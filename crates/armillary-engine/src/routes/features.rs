use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::json;

static FEATURES_TOML: &str = include_str!("../../../../features.toml");

#[derive(Debug, Deserialize, Serialize)]
struct FeatureEntry {
    name: String,
    landed: String,
    commit: String,
    summary: String,
}

#[derive(Debug, Deserialize)]
struct FeaturesManifest {
    features: Vec<FeatureEntry>,
}

pub async fn features() -> Json<serde_json::Value> {
    let manifest: FeaturesManifest = toml::from_str(FEATURES_TOML)
        .unwrap_or(FeaturesManifest { features: vec![] });
    Json(json!({
        "commit": env!("GIT_COMMIT"),
        "features": manifest.features,
    }))
}
