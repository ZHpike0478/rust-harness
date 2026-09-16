// SPDX-License-Identifier: MIT OR Apache-2.0
//
// `#[tool]` proc-macro for `llama-harness`.
//
// Turns:
//
//     #[tool(description = "Get weather for a city")]
//     async fn get_weather(city: String, units: TempUnits) -> Weather {
//         ...
//     }
//
// Into a struct that implements `llama_harness::Tool`. The struct is
// generic over the function's argument types via `schemars`, so the
// generated `parameters()` method returns a real JSON Schema for the
// fn's argument tuple.
//
// Usage:
//
//     llama_harness::Harness::builder()
//         .register_tool(get_weather::Tool::new())
//         ...
//
// The generated struct is named `<fn_name>::Tool` to avoid clashing
// with the function itself.

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{quote, ToTokens};
use syn::{
    parse_macro_input, FnArg, ItemFn, Pat, ReturnType, Type,
};

/// Attribute macro: turn an async fn into a `Tool` impl.
///
/// Accepted attributes:
///   * `description = "..."` (required)
///   * `name = "..."` (optional, defaults to fn name)
///
/// The macro inspects the function's argument list. Each argument must
/// be a typed parameter that `schemars::schema_for!` can introspect.
/// The return type becomes the tool's success value.
#[proc_macro_attribute]
pub fn tool(args: TokenStream, input: TokenStream) -> TokenStream {
    // Parse attributes (description, optional name).
    let attr_args = parse_macro_input!(args as ToolArgs);

    // Parse the function.
    let mut func = parse_macro_input!(input as ItemFn);

    if func.sig.asyncness.is_none() {
        return syn::Error::new_spanned(
            func.sig.fn_token,
            "`#[tool]` requires the function to be `async`",
        )
        .to_compile_error()
        .into();
    }

    let fn_name = &func.sig.ident;
    let fn_vis = &func.vis;
    let fn_name_str = attr_args
        .name
        .unwrap_or_else(|| fn_name.to_string());
    let description = attr_args.description.unwrap_or_else(|| String::new());
    if description.is_empty() {
        return syn::Error::new(
            Span::call_site(),
            "`#[tool]` requires `description = \"...\"`",
        )
        .to_compile_error()
        .into();
    }

    // Strip the original `async` keyword so we can call the function in
    // a sync context (the generated impl wraps it in a tokio block).
    func.sig.asyncness = None;

    // Build a "(arg1_ty, arg2_ty, ...)" tuple type for schemars to introspect.
    let arg_tys: Vec<&Type> = func
        .sig
        .inputs
        .iter()
        .filter_map(|a| match a {
            FnArg::Typed(pt) => Some(&*pt.ty),
            FnArg::Receiver(_) => None,
        })
        .collect();

    let arg_pats: Vec<&Pat> = func
        .sig
        .inputs
        .iter()
        .filter_map(|a| match a {
            FnArg::Typed(pt) => Some(&*pt.pat),
            FnArg::Receiver(_) => None,
        })
        .collect();

    let ret_ty = match &func.sig.output {
        ReturnType::Default => quote! { () },
        ReturnType::Type(_, t) => quote! { #t },
    };

    let struct_name = quote::format_ident!("Tool");
    let call_struct = quote::format_ident!("ToolCall");

    // The generated struct is generic over the argument tuple type.
    let arg_tuple_ty = if arg_tys.is_empty() {
        quote! { () }
    } else {
        quote! { ( #( #arg_tys ),* ) }
    };

    let expanded = quote! {
        // Keep the original function definition.
        #fn_vis #func

        // The generated struct + Tool impl.
        #[derive(Default)]
        pub struct #struct_name {
            _priv: (),
        }

        impl #struct_name {
            pub fn new() -> Self {
                Self { _priv: () }
            }
        }

        // Marker so the user can refer to the call site.
        #[doc(hidden)]
        pub struct #call_struct;

        #[async_trait::async_trait]
        impl llama_harness::Tool for #struct_name {
            fn name(&self) -> &str { #fn_name_str }
            fn description(&self) -> &str { #description }

            fn parameters(&self) -> serde_json::Value {
                use schemars::JsonSchema;
                // Generate a JSON Schema for a tuple of the fn's argument types.
                // We synthesize a transient type alias so we can call schemars::schema_for!.
                type __Args = #arg_tuple_ty;
                let mut s = schemars::schema_for!(__Args);
                // Strip schemars' $ref to root; we want an object schema.
                if let Some(o) = s.get_mut("$ref") {
                    *o = serde_json::Value::String("#/definitions/Args".into());
                }
                serde_json::to_value(s).unwrap_or(serde_json::json!({"type": "object"}))
            }

            async fn execute(
                &self,
                args: serde_json::Value,
                _ctx: llama_harness::ToolContext,
            ) -> llama_harness::Result<llama_harness::ToolOutput> {
                // Deserialize the JSON object into the tuple.
                let parsed: #arg_tuple_ty = serde_json::from_value(args)
                    .map_err(|e| llama_harness::HarnessError::MalformedToolCall(
                        format!("bad arguments: {}", e)
                    ))?;
                // Call the fn. The user's signature was `async fn`, but we
                // stripped asyncness, so it's a regular fn returning a future.
                let fut = #fn_name(parsed);
                let result: #ret_ty = fut.await;
                Ok(llama_harness::ToolOutput::data(
                    serde_json::to_value(&result)
                        .unwrap_or(serde_json::Value::Null)
                ))
            }
        }
    };

    expanded.into()
}

/// Parsed attribute arguments for `#[tool]`.
struct ToolArgs {
    name: Option<String>,
    description: Option<String>,
}

impl Parse for ToolArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut name = None;
        let mut description = None;
        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            input.parse::<syn::Token![=]>()?;
            let val: syn::LitStr = input.parse()?;
            match key.to_string().as_str() {
                "name" => name = Some(val.value()),
                "description" => description = Some(val.value()),
                _ => {
                    return Err(syn::Error::new_spanned(
                        key,
                        "unknown #[tool] attribute (expected `name` or `description`)",
                    ));
                }
            }
            if input.peek(syn::Token![,]) {
                let _: syn::Token![,] = input.parse()?;
            }
        }
        Ok(Self { name, description })
    }
}

// We need Parse in scope. The use below would normally import it from syn,
// but it's already used via parse_macro_input!.
use syn::parse::Parse;
