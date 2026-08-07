use crate::models;
use crate::provider::{choose_provider, ProviderChoice};
use crate::state::SharedState;
use axum::{extract::State, Json};
use serde::Serialize;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelEntry {
    pub id: String,
    pub label: Option<String>,
    /// `anthropic` | `zen` — `choose_provider`'s answer, so the app never
    /// re-derives the prefix rule.
    pub provider: String,
    /// Whether this host holds a key for that provider.
    ///
    /// **Advisory only.** The engine does not enforce it: create accepts
    /// any model (design decision 3), and an instance pinned to an
    /// unusable one fails its first turn with `no_api_key` on an event
    /// that names the model. This exists so the picker can grey a row out
    /// and say why, not so it can refuse.
    pub usable: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsResponse {
    pub default: Option<String>,
    pub models: Vec<ModelEntry>,
}

pub async fn models(State(state): State<SharedState>) -> Json<ModelsResponse> {
    let catalog = models::load(&state.models_path);
    let entries = catalog
        .models
        .into_iter()
        .map(|m| {
            let (provider, usable) = match choose_provider(&m.id) {
                ProviderChoice::Anthropic => ("anthropic", state.anthropic_key_present),
                ProviderChoice::Zen { .. } => ("zen", state.zen_key_present),
            };
            ModelEntry {
                id: m.id,
                label: m.label,
                provider: provider.to_string(),
                usable,
            }
        })
        .collect();
    Json(ModelsResponse { default: catalog.default, models: entries })
}
