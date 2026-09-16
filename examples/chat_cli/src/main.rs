//! chat_cli -- interactive REPL for llama-harness.
//!
//! Usage:
//!
//!     cargo run -p chat_cli --
//!     cargo run -p chat_cli -- --registry C:/path/to/models.toml
//!
//! Commands (slash-prefixed):
//!
//!     /list                  list all known model names
//!     /loaded                list currently-loaded models
//!     /use <name>            switch the active model
//!     /load <name>           load a model from the registry
//!     /unload <name>         unload a model
//!     /info [name]           show info about active model (or named)
//!     /system <text>         set/clear the system prompt
//!     /temp <float>          set temperature
//!     /help                  show help
//!     /quit                  exit
//!
//! Anything else is treated as a user prompt.

use llama_harness::{
    ChatRequest, Harness, Message, Role, SamplingParams, StreamEvent,
};

use futures::StreamExt;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,llama_harness=debug".into()),
        )
        .with_target(false)
        .init();

    // Parse args (very simple -- only --registry).
    let mut registry_path: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--registry" {
            if let Some(p) = args.next() {
                registry_path = Some(PathBuf::from(p));
            }
        }
    }

    let mut builder = Harness::builder().max_loaded_models(2);
    if let Some(p) = registry_path {
        builder = builder.registry_path(p);
    }
    let harness = builder.build().await?;

    println!("llama-harness chat_cli -- stub backend");
    println!("known models: {:?}", harness.list());
    println!();
    println!("commands: /help");
    println!();

    let ctx = Arc::new(tokio::sync::Mutex::new(Context::default()));
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush()?;
        let mut line = String::new();
        let n = stdin.lock().read_line(&mut line)?;
        if n == 0 {
            println!();
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if line.starts_with('/') {
            match handle_command(line, &harness, ctx.clone()).await? {
                CommandOutcome::Continue => continue,
                CommandOutcome::Quit => break,
            }
        } else {
            // User prompt -> chat.
            let mut c = ctx.lock().await;
            c.messages.push(Message::user(line));

            let req = ChatRequest {
                messages: c.messages.clone(),
                model: None,
                tools: Default::default(),
                tool_execution: Default::default(),
                max_tool_iterations: 4,
                sampling: c.sampling.clone(),
                stop: vec!["</s>".to_string(), "<|eot|>".to_string()],
            };

            let active = harness.active_model().await;
            let model_name = match active.as_deref() {
                Some(n) => n.to_string(),
                None => {
                    println!("(no active model -- try `/load <name>` first)");
                    continue;
                }
            };
            drop(c);

            print!("[{model_name}] ");
            stdout.flush()?;
            let mut stream = harness.chat_stream(req);
            let mut assistant_text = String::new();
            let mut last = None;
            while let Some(ev) = stream.next().await {
                match ev {
                    StreamEvent::Token { text } => {
                        print!("{text}");
                        stdout.flush()?;
                        assistant_text.push_str(&text);
                    }
                    StreamEvent::Done(resp) => {
                        last = Some(resp);
                        break;
                    }
                    StreamEvent::Error(msg) => {
                        eprintln!("\nerror: {msg}");
                    }
                    _ => {}
                }
            }
            println!();

            if let Some(resp) = last {
                let mut c = ctx.lock().await;
                c.messages.push(Message::assistant(resp.final_text.clone()));
                // Token ledger
                c.prompt_tokens += resp.prompt_tokens as u64;
                c.completion_tokens += resp.completion_tokens as u64;
                eprintln!(
                    "[tokens: prompt={} completion={} total={}]",
                    resp.prompt_tokens,
                    resp.completion_tokens,
                    c.prompt_tokens + c.completion_tokens,
                );
            }
        }
    }

    Ok(())
}

#[derive(Default)]
struct Context {
    system_prompt: Option<String>,
    messages: Vec<Message>,
    sampling: SamplingParams,
    prompt_tokens: u64,
    completion_tokens: u64,
}

enum CommandOutcome {
    Continue,
    Quit,
}

async fn handle_command(
    line: &str,
    h: &Harness,
    ctx: Arc<tokio::sync::Mutex<Context>>,
) -> anyhow::Result<CommandOutcome> {
    let mut parts = line.split_whitespace();
    let cmd = parts.next().unwrap();
    match cmd {
        "/help" => {
            println!("commands:");
            println!("  /list                list all known model names");
            println!("  /loaded              list currently-loaded models");
            println!("  /use <name>          switch active model");
            println!("  /load <name>         load a model from the registry");
            println!("  /unload <name>       unload a model");
            println!("  /info [name]         show info about active model (or named)");
            println!("  /system <text...>    set system prompt (empty to clear)");
            println!("  /temp <float>        set temperature (0.0 - 2.0)");
            println!("  /clear               clear conversation history");
            println!("  /tokens              show cumulative token usage");
            println!("  /quit, /exit         exit");
        }
        "/list" => {
            let names = h.list();
            println!("known models ({}):", names.len());
            for n in names {
                println!("  {n}");
            }
        }
        "/loaded" => {
            let loaded = h.loaded_models().await;
            if loaded.is_empty() {
                println!("(none)");
            } else {
                for m in loaded {
                    println!(
                        "  {} -- {} ({}B params)",
                        m.id, m.family, m.n_params_b
                    );
                }
            }
        }
        "/use" => {
            if let Some(name) = parts.next() {
                h.set_active(name).await?;
                println!("active model: {name}");
            } else {
                eprintln!("usage: /use <name>");
            }
        }
        "/load" => {
            if let Some(name) = parts.next() {
                let info = h.load(name).await?;
                println!(
                    "loaded: {} -- {} ({}B params)",
                    info.id, info.family, info.n_params_b
                );
            } else {
                eprintln!("usage: /load <name>");
            }
        }
        "/unload" => {
            if let Some(name) = parts.next() {
                h.unload(name).await?;
                println!("unloaded: {name}");
            } else {
                eprintln!("usage: /unload <name>");
            }
        }
        "/info" => {
            if let Some(name) = parts.next() {
                let loaded = h.loaded_models().await;
                match loaded.iter().find(|m| m.id == name) {
                    Some(m) => println!("{m:?}"),
                    None => println!("(not loaded)"),
                }
            } else {
                println!("active: {:?}", h.active_model().await);
            }
        }
        "/system" => {
            let rest: String = parts.collect::<Vec<_>>().join(" ");
            let mut c = ctx.lock().await;
            c.system_prompt = if rest.is_empty() { None } else { Some(rest.clone()) };
            let sys_clone = c.system_prompt.clone();
            // Rebuild message history: drop everything and prepend new system.
            c.messages.clear();
            if let Some(sys) = c.system_prompt.clone() {
                c.messages.push(Message::system(sys));
            }
            println!("system prompt set.");
        }
        "/temp" => {
            if let Some(t) = parts.next() {
                let t: f32 = t.parse()?;
                let mut c = ctx.lock().await;
                c.sampling.temperature = t;
                println!("temperature: {t}");
            } else {
                eprintln!("usage: /temp <float>");
            }
        }
        "/clear" => {
            let mut c = ctx.lock().await;
            c.messages.clear();
            if let Some(sys) = c.system_prompt.clone() {
                c.messages.push(Message::system(sys));
            }
            println!("cleared.");
        }
        "/tokens" => {
            let c = ctx.lock().await;
            println!(
                "prompt={} completion={} total={}",
                c.prompt_tokens,
                c.completion_tokens,
                c.prompt_tokens + c.completion_tokens
            );
        }
        "/quit" | "/exit" => return Ok(CommandOutcome::Quit),
        other => eprintln!("unknown command: {other}  (try /help)"),
    }
    Ok(CommandOutcome::Continue)
}
