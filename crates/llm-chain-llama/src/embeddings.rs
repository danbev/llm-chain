use crate::context::ContextParams;
use crate::context::LLamaContext;
use crate::options::{LlamaInvocation, DEFAULT_OPTIONS};
use crate::tokenizer;
use async_trait::async_trait;
use futures::future::try_join_all;
use llm_chain::options::{options_from_env, Opt, OptDiscriminants, Options, OptionsCascade};
use llm_chain::prompt::Data;
use llm_chain::traits::EmbeddingsCreationError;
use llm_chain::traits::{self, EmbeddingsError};
use std::sync::Arc;
use std::{error::Error, fmt::Debug};
use tokio::sync::Mutex;

/// Generate embeddings using the llama.
///
/// This intended be similar to running the embedding example in llama.cpp:
/// ./embedding -m <path_to_model> --log-disable -p "Hello world" 2>/dev/null
///
pub struct Embeddings {
    context: Arc<Mutex<LLamaContext>>,
    options: Options,
}

#[async_trait]
impl traits::Embeddings for Embeddings {
    type Error = LlamaEmbeddingsError;

    async fn embed_texts(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, Self::Error> {
        let futures = texts.into_iter().map(|text| self.embed_query(text));
        let embeddings = try_join_all(futures).await?;
        Ok(embeddings)
    }

    async fn embed_query(&self, query: String) -> Result<Vec<f32>, Self::Error> {
        let options = vec![&DEFAULT_OPTIONS, &self.options];
        let invocation =
            LlamaInvocation::new(OptionsCascade::from_vec(options), &Data::Text(query)).unwrap();
        let embeddings = self.get_embeddings(invocation).await?;
        Ok(embeddings)
    }
}

#[allow(dead_code)]
impl Embeddings {
    pub fn new_with_options(options: Options) -> Result<Self, EmbeddingsCreationError> {
        let max_context_size = if let Some(Opt::MaxContextSize(max_context_size)) =
            options.get(OptDiscriminants::MaxContextSize)
        {
            max_context_size
        } else {
            &2048
        };

        let mut context_params = ContextParams::new();
        context_params.n_ctx = *max_context_size as i32;
        context_params.embedding = true;
        Ok(Self {
            context: Arc::new(Mutex::new(LLamaContext::from_file_and_params(
                Self::get_model_path(&options)?.as_str(),
                Some(&context_params),
            )?)),
            options,
        })
    }

    fn get_model_path(options: &Options) -> Result<String, EmbeddingsCreationError> {
        let opts_from_env =
            options_from_env().map_err(|err| EmbeddingsCreationError::InnerError(err.into()))?;
        let cas = OptionsCascade::new()
            .with_options(&DEFAULT_OPTIONS)
            .with_options(&opts_from_env)
            .with_options(&options);
        let model_path = cas
            .get(OptDiscriminants::Model)
            .and_then(|x| match x {
                Opt::Model(m) => Some(m),
                _ => None,
            })
            .ok_or(EmbeddingsCreationError::FieldRequiredError(
                "model_path".to_string(),
            ))?;
        Ok(model_path.to_path())
    }

    async fn get_embeddings(
        &self,
        input: LlamaInvocation,
    ) -> Result<Vec<f32>, LlamaEmbeddingsError> {
        let context = self.context.clone();
        let embeddings = tokio::task::spawn_blocking(move || {
            let context = context.blocking_lock();
            let prompt_text = input.prompt.to_text();
            let tokenized_input = tokenizer::tokenize(&context, prompt_text.as_str(), true);
            let _ = context
                .llama_eval(
                    tokenized_input.as_slice(),
                    tokenized_input.len() as i32,
                    0,
                    &input,
                )
                .map_err(|e| LlamaEmbeddingsError::InnerError(e.into()));
            context.llama_get_embeddings()
        });
        embeddings
            .await
            .map_err(|e| LlamaEmbeddingsError::InnerError(e.into()))
    }
}

#[derive(thiserror::Error, Debug)]
pub enum LlamaEmbeddingsError {
    #[error("error when trying to generate embeddings: {0}")]
    InnerError(#[from] Box<dyn Error + Send + Sync>),
}

impl EmbeddingsError for LlamaEmbeddingsError {}
