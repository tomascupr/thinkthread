# Configuration Reference

This page documents the ThinkThreadConfig class that manages the configuration for the ThinkThread SDK.

## ThinkThreadConfig

::: thinkthread_sdk.config.ThinkThreadConfig
    options:
      show_bases: false
      members:
        - openai_api_key
        - anthropic_api_key
        - hf_api_token
        - provider
        - openai_model
        - anthropic_model
        - hf_model
        - alternatives
        - rounds
        - max_rounds
        - use_pairwise_evaluation
        - use_self_evaluation
        - parallel_alternatives
        - parallel_evaluation
        - use_caching
        - early_termination
        - early_termination_threshold
        - concurrency_limit
        - enable_monitoring
        - use_batched_requests
        - use_fast_similarity
        - use_adaptive_temperature
        - initial_temperature
        - generation_temperature
        - min_generation_temperature
        - temperature_decay_rate
        - prompt_dir

## Helper Functions

::: thinkthread_sdk.config.create_config
    options:
      show_source: false
