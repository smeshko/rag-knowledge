"""Application settings loaded from environment and `.env`.

Required fields (`database_url`, `redis_url`, `openai_api_key`) have no
default — instantiating `Settings()` raises a `pydantic.ValidationError`
that names the missing field if they aren't provided. Every other field
has a documented default that mirrors `.env.example`.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from redis.asyncio.connection import parse_url as redis_parse_url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    database_url: str
    redis_url: str
    redis_password: str = ""
    openai_api_key: str

    # Provider switch for the extraction + answer LLM (Epic 19.1). Default
    # "openai" keeps unset config byte-for-byte today's behaviour; "anthropic"
    # routes both factories to Claude. Validated against the supported set, and a
    # model_validator requires anthropic_api_key when "anthropic" is selected.
    llm_provider: str = "openai"
    # Anthropic credentials/model/token cap. anthropic_api_key is genuinely
    # optional (only required when llm_provider == "anthropic"); the rate-limit
    # retry / request-timeout knobs are shared with OpenAI (llm_*). max_tokens is
    # required by Anthropic's Messages API and bounds extraction output (ge=1);
    # the 8192 default stays under the SDK's ~16K non-streaming timeout guard.
    anthropic_api_key: str | None = None
    anthropic_llm_model: str = "claude-sonnet-4-6"
    anthropic_max_tokens: int = Field(default=8192, ge=1)
    # Batch submission (Epic 19.2). The cron submitter chunks pending windows by
    # BOTH a request count and an estimated serialized-bytes cap (request_input +
    # repeated schema can blow past 256MB well under the count cap). max_requests
    # default is lowered from Anthropic's 100k so the byte cap is usually the
    # binding limit; max_bytes keeps a margin under the 256MB hard cap.
    anthropic_batch_max_requests: int = Field(default=10000, ge=1, le=100000)
    anthropic_batch_max_bytes: int = Field(default=200_000_000, ge=1, le=256_000_000)
    anthropic_batch_submit_interval_minutes: int = Field(default=5, ge=1)
    # How long a SUBMITTING batch may sit before reconciliation treats it as a
    # crashed submit and reverts it to PENDING for dedup-safe re-submission
    # (DECISIONS #7; review #2.1).
    anthropic_batch_submitting_timeout_minutes: int = Field(default=60, ge=1)
    # Batch poll/ingest (Epic 19.3). The poller cron interval, and the per-window
    # cap on re-submitting expired/transient-errored windows (at the cap the window
    # becomes a REJECTED audit run rather than retrying forever).
    anthropic_batch_poll_interval_minutes: int = Field(default=5, ge=1)
    anthropic_batch_max_submit_attempts: int = Field(default=2, ge=1)

    # DeepSeek (Epic 23.4) — OpenAI-compatible transport on its own endpoint, so
    # it reuses OpenAILLMProvider rather than adding a client. deepseek_api_key is
    # only required when selected (the registry-driven cross-field validator
    # enforces that). The model id and base URL are settings, not constants,
    # because both were established from a dated docs check rather than from a
    # live call — Phase 23.5 may need to retarget without a code change.
    deepseek_api_key: str | None = None
    deepseek_llm_model: str = "deepseek-v4-pro"
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    # claude_cli — the local `claude` binary in -p mode, drawing claude.ai
    # subscription quota instead of API billing. No key field: auth is the CLI's
    # own login (a missing login fails loudly at the first call). The model is
    # the pinned full name, never the `opus` alias — it is part of the extraction
    # cache key and every ExtractionRun row, and an alias would let a CLI update
    # change model vintage under a stable key. The timeout kills the subprocess
    # and raises LLMTechnicalError (Opus windows can be slow, hence 300s); the
    # binary path is overridable for non-PATH installs.
    claude_cli_model: str = "claude-opus-5"
    claude_cli_timeout_seconds: float = Field(default=300.0, ge=1)
    claude_cli_binary: str = "claude"

    # Judge provider/model for the eval harness (Epic 23.3). Both None by default,
    # meaning "the judge runs on the same provider as the model under test" — i.e.
    # today's behaviour. Setting them is what makes a cross-provider comparison
    # meaningful: with one provider serving both roles, every candidate grades
    # itself. `None` rather than a concrete default on purpose, so an Anthropic
    # deployment's behaviour does not silently change the moment this ships.
    judge_llm_provider: str | None = None
    judge_llm_model: str | None = None

    llm_model: str = "gpt-4.1"
    # Retarget the OpenAI SDK at any OpenAI-*compatible* endpoint (Epic 23.4).
    # None keeps the SDK's own default (api.openai.com). This is transport only:
    # the identity label a provider records is set alongside it by the provider
    # registry, because that label is written to every ExtractionRun row and is
    # part of the extraction cache key — pointing the client elsewhere without
    # changing the label would file another vendor's runs under "openai" and let
    # the two satisfy each other's cache lookups.
    llm_base_url: str | None = None
    # How schema-constrained output is requested (Epic 23.4). None means "let the
    # selected provider's registry entry choose" — a non-None default could not
    # express that, and would let llm_provider=deepseek + json_schema (a guaranteed
    # 400, since DeepSeek's response_format accepts only text/json_object) load
    # clean and fail at the first request. See providers.llm.openai for the values.
    llm_structured_output_mode: str | None = None
    # The identity label runs against llm_base_url are recorded under. Required
    # whenever llm_base_url is set (enforced below): the label is written to every
    # ExtractionRun row and is part of the extraction cache key, so retargeting the
    # URL while leaving the label at "openai" would file another vendor's runs as
    # OpenAI's and let the two satisfy each other's cache lookups.
    llm_provider_label: str | None = None
    # Phase 9.5: bound the provider's rate-limit retry loop and per-request
    # timeout. retries=0 disables retries (raise on the first 429); the timeout
    # is a float so it feeds chat.completions.create(timeout=…) without a cast.
    llm_max_rate_limit_retries: int = Field(default=5, ge=0)
    llm_request_timeout_seconds: float = Field(default=60.0, ge=1)
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = Field(default=100, ge=1, le=2048)

    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_enabled: bool = False

    personal_api_token: str | None = None
    debug_endpoints_enabled: bool = False

    local_storage_root: str = "./data/storage"

    pdf_window_size_pages: int = 3
    pdf_overlap_pages: int = 1
    pdf_min_text_chars_for_page: int = 20

    # Identity of the production PDF text extractor (method + build), stamped onto
    # every SourceSpan a run writes so the `auto` reprocess selector (Epic 11.2) can
    # detect whether the extractor changed since a version was produced. Two PyMuPDF
    # builds both stamp extraction_method="embedded_text" but should differ here if
    # the build changed.
    pdf_text_extractor: str = "pymupdf:embedded_text"

    # Phase 9.5: windows per per-batch commit in the extraction loop. A crash
    # rolls back the in-flight batch, so at most batch_size − 1 windows of
    # OpenAI spend are repeated on resume (DECISIONS #4).
    extraction_commit_batch_size: int = Field(default=5, ge=1)
    # How many windows inside one commit batch may have a provider call in flight
    # at once. 1 preserves the original strictly-sequential loop exactly.
    #
    # Only the provider call fans out; every database write stays sequential on
    # the batch's single session (an AsyncSession is not concurrency-safe). The
    # ceiling is therefore extraction_commit_batch_size — raise both together.
    #
    # Measured against the claude CLI: 4 concurrent calls sharing one cwd
    # completed in the wall time of one (4.1s vs 3.6s solo) and every one of them
    # READ the shared prompt cache (3,608 tokens) rather than re-creating it, so
    # concurrency does not forfeit the cache win. The real cost is burst rate:
    # N concurrent calls drain a subscription quota N times faster, and on a
    # provider with no retry loop, exhaustion fails the document.
    extraction_max_concurrent_windows: int = Field(default=1, ge=1)

    # ge=1 so the answer route's effective-limit fallback (and search's own clamp)
    # is guaranteed positive even with a bad env value — a ≤0 default would poison
    # the context-pack item cap (Epic 17.2, review #3).
    search_default_limit: int = Field(default=10, ge=1)
    search_keyword_top_k: int = 50
    search_vector_top_k: int = 50
    search_rrf_k: int = 60
    # Per-leg RRF source weights and the grouping supporting-chunk bonus (doc 7 § 8/§ 9).
    keyword_source_weight: float = Field(default=1.0, ge=0)
    vector_source_weight: float = Field(default=1.0, ge=0)
    search_supporting_chunk_bonus: float = Field(default=0.05, ge=0)
    search_supporting_chunk_bonus_cap: float = Field(default=0.15, ge=0)
    # The provider-name half of the (embedding_provider, embedding_model) vector-leg
    # filter — embedding_model already exists; the EmbeddingProvider ABC carries no
    # name, so the search facade sources both from Settings (doc 7; DECISIONS #6).
    embedding_provider: str = "openai"

    # Chunk-type boosts (doc 7 § 7) — the full per-side tables. Keyword favours
    # title/ingredients; vector favours summary/full. Missing types default to 1.0
    # at merge time.
    recipe_keyword_boost_title: float = 1.40
    recipe_keyword_boost_ingredients: float = 1.20
    recipe_keyword_boost_steps: float = 1.05
    recipe_keyword_boost_summary: float = 1.00
    recipe_keyword_boost_full: float = 0.95
    recipe_vector_boost_summary: float = 1.20
    recipe_vector_boost_full: float = 1.10
    recipe_vector_boost_steps: float = 1.00
    recipe_vector_boost_ingredients: float = 0.95
    recipe_vector_boost_title: float = 0.90

    worker_max_jobs: int = Field(default=1, ge=1)
    worker_job_timeout_seconds: int = Field(default=600, ge=1)
    # Per-function timeout for `process_document` only, kept separate from the
    # worker-wide default because the two bound very different work. A document
    # is a whole book: windows are extracted sequentially, and a
    # subscription-quota provider (claude_cli) spends ~35s per window against a
    # ~pages/2 window count, so a 250-page cookbook runs over an hour. The 600s
    # default would cancel it, and arq retries CancelledError up to max_tries, so
    # such a book died as `max 3 retries exceeded` after ~30 minutes of progress.
    #
    # Liveness is NOT delegated to this timeout: sweep_stuck_jobs reaps on the
    # per-batch `last_progress_at` heartbeat (every extraction_commit_batch_size
    # windows), so a wedged document is still marked FAILED within
    # stuck_job_timeout_minutes no matter how generous this is. This only bounds
    # the arq task itself.
    #
    # Cost of a large value: arq derives its in-progress lock TTL from the
    # largest registered timeout, so after a hard worker crash re-delivery of any
    # job waits out that TTL. The document-level sweep plus
    # POST /documents/{id}/reprocess cover that window; raise deliberately.
    document_job_timeout_seconds: int = Field(default=14400, ge=1)
    worker_keep_result_seconds: int = Field(default=60, ge=0)
    worker_health_check_interval_seconds: int = Field(default=30, ge=1)

    stuck_job_timeout_minutes: int = Field(default=30, ge=1)
    stuck_job_check_interval_minutes: int = Field(default=5, ge=1, le=60)
    # Phase 21.3 (plan D1b): items stuck in `indexing` past this threshold are
    # returned to `needs_review` by the item-level pass in sweep_stuck_jobs.
    # Deliberately shorter than stuck_job_timeout_minutes — a single-item index
    # job is one chunk+embed round-trip, not a multi-window extraction.
    stuck_indexing_timeout_minutes: int = Field(default=15, ge=1)

    # Soft-validation thresholds (Epic 9 Phase 9.3, doc 4 § Soft validation).
    # Review heuristics, not calibrated truth — a candidate below a confidence
    # floor or outside the char band is persisted as needs_review, not dropped.
    extraction_min_overall_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    extraction_min_boundary_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    extraction_min_normalization_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    extraction_min_recipe_chars: int = Field(default=200, ge=0)
    extraction_max_recipe_chars: int = Field(default=20000, ge=1)
    # Bounds on the *assembly* recipe exemption (see
    # ``validation._is_assembly_recipe``): a candidate with no steps, at most
    # this many ingredients and a body no longer than this is judged method-free
    # by design — the bowl-cookbook genre prints such recipes as a title plus a
    # list of components documented elsewhere — and is spared ``no_steps`` /
    # ``recipe_too_short``. Sized from the two ingested books: every real
    # assembly recipe held 3-10 ingredients in 96-260 characters, while the
    # failure mode these rules exist to catch (a recipe truncated at a window
    # boundary, keeping its ingredients and losing its method) held 17-32.
    # The gap is wide; these sit inside it. The floor exists to exclude the
    # other short step-less shape — a recipe whose method was written as prose
    # into body_text and never structured; that is a real defect and must keep
    # flagging.
    extraction_assembly_min_ingredients: int = Field(default=3, ge=1)
    extraction_assembly_max_ingredients: int = Field(default=12, ge=1)
    extraction_assembly_max_chars: int = Field(default=400, ge=1)

    # Query-time answer layer (Epic 17, doc 8 § 3). answer_llm_model is left None
    # and resolved to llm_model at the dependency boundary (a class default can't
    # reference a sibling field). The two cap fields are Field(ge=1) so a ≤0 env
    # value is rejected at Settings load rather than silently including nearly all
    # results (negative) or forcing an empty context (zero).
    answer_llm_model: str | None = None
    answer_prompt_version: str = "answer-recommendation-v1"
    answer_schema_version: str = "answer.v1"
    answer_context_item_limit: int = Field(default=6, ge=1)
    answer_matched_chunks_per_item: int = Field(default=3, ge=1)

    # Reranking (Epic 18). Off by default; the rerank step is wired into search() in
    # Phase 18.2. rerank_top_n is bounded (ge=1, le=200) so a ≤0 value can't silently
    # disable reranking and a huge value can't feed an LLM reranker a costly fan-out.
    reranking_enabled: bool = False
    rerank_provider: str = "openai"
    rerank_model: str = "gpt-4.1"
    rerank_top_n: int = Field(default=50, ge=1, le=200)
    # Hot-path budget (Epic 18.2): reranking is on the synchronous search path and
    # Chunk.text is unbounded, so cap the per-candidate payload and run the reranker
    # under a short timeout (not the 60s ingestion default) — a slow rerank times out
    # to RerankerTechnicalError → baseline fallback rather than stalling search.
    rerank_max_chars_per_candidate: int = Field(default=2000, ge=1)
    rerank_request_timeout_seconds: float = Field(default=8.0, gt=0)

    @field_validator("redis_url")
    @classmethod
    def _redis_url_requires_credentials(cls, value: str) -> str:
        # Phase 1.6: reject DSNs without credentials so a half-migrated .env
        # (REDIS_PASSWORD added but REDIS_URL not updated) fails at Settings
        # load instead of silently hitting NOAUTH at runtime. Delegated to
        # redis-py's parser so every scheme the client accepts
        # (redis://, rediss://, unix://) stays valid here too.
        try:
            parsed = redis_parse_url(value)
        except ValueError as exc:
            raise ValueError(f"REDIS_URL is not a valid Redis DSN: {exc}") from exc
        if not parsed.get("password"):
            raise ValueError("REDIS_URL must include credentials, e.g. redis://:pwd@host:port/db")
        return value

    @model_validator(mode="after")
    def _redis_password_matches_url(self) -> Settings:
        # Phase 1.6: catch the mismatch case (REDIS_PASSWORD updated but the
        # password in REDIS_URL drifted, or vice versa) — would otherwise pass
        # Settings load and explode with WRONGPASS at the first arq write.
        # Skipped when REDIS_PASSWORD is unset so envs that authenticate via
        # a fully-credentialed REDIS_URL alone stay supported. parse_url
        # decodes percent-encoded passwords, so the comparison is direct.
        if not self.redis_password:
            return self
        url_password = redis_parse_url(self.redis_url).get("password")
        if url_password != self.redis_password:
            raise ValueError(
                "REDIS_URL password does not match REDIS_PASSWORD; update both in .env"
            )
        return self

    @field_validator("llm_provider")
    @classmethod
    def _llm_provider_supported(cls, value: str) -> str:
        # Reject an unsupported provider at load so a typo (e.g. "gemini") fails
        # fast rather than falling through to the OpenAI branch at runtime. The
        # allow-list is the provider registry itself (Epic 23.4), so registering a
        # provider does not also require editing a literal set here.
        from rag_recipes.providers.llm.registry import supported_providers

        supported = supported_providers()
        if value not in supported:
            raise ValueError(f"llm_provider must be one of {sorted(supported)}; got {value!r}")
        return value

    @field_validator("judge_llm_provider")
    @classmethod
    def _judge_llm_provider_supported(cls, value: str | None) -> str | None:
        # Same registry allow-list as llm_provider, but None is meaningful here:
        # it is the "judge on the extraction provider" sentinel, not an absence.
        from rag_recipes.providers.llm.registry import supported_providers

        if value is None:
            return value
        supported = supported_providers()
        if value not in supported:
            raise ValueError(
                f"judge_llm_provider must be one of {sorted(supported)}; got {value!r}"
            )
        return value

    @field_validator("llm_structured_output_mode")
    @classmethod
    def _structured_output_mode_supported(cls, value: str | None) -> str | None:
        # None is the "defer to the registry entry" sentinel and is always valid.
        # A typo'd mode must fail at load: an unknown value would otherwise reach
        # the provider and produce a request shape the vendor rejects mid-run.
        from rag_recipes.providers.llm.openai import STRUCTURED_OUTPUT_MODES

        if value is not None and value not in STRUCTURED_OUTPUT_MODES:
            raise ValueError(
                "llm_structured_output_mode must be one of "
                f"{sorted(STRUCTURED_OUTPUT_MODES)}; got {value!r}"
            )
        return value

    @model_validator(mode="after")
    def _provider_api_key_required_when_selected(self) -> Settings:
        # A class default can't reference a sibling field, so the cross-field rule
        # (key required when the provider needing it is selected) lives here. Fails
        # at Settings load with a message naming the missing field rather than at
        # the first call. Generic over the registry since Epic 23.4, but the message
        # is byte-identical to the pre-registry Anthropic-specific one — it is
        # user-facing config feedback, and that refactor changed no behaviour.
        # api_key_field is None for providers whose key is already unconditionally
        # required (openai), so this never fires on an empty-string openai_api_key.
        # Note this does NOT narrow the key to str for mypy; the registry factories
        # still narrow str | None → str before use.
        from rag_recipes.providers.llm.registry import get_spec

        field = get_spec(self.llm_provider).api_key_field
        if field is not None and not getattr(self, field):
            raise ValueError(f"{field} is required when llm_provider == {self.llm_provider!r}")
        # Epic 23.3: the judge may run on a different provider, which needs its own
        # key. Checked here rather than at the first judge call because that call
        # happens *after* extraction has already spent money on the run.
        judge_provider = self.judge_llm_provider
        if judge_provider is not None:
            judge_field = get_spec(judge_provider).api_key_field
            if judge_field is not None and not getattr(self, judge_field):
                raise ValueError(
                    f"{judge_field} is required when judge_llm_provider == {judge_provider!r}"
                )
        return self

    @model_validator(mode="after")
    def _structured_output_mode_serveable_by_provider(self) -> Settings:
        # A mode the selected vendor's API does not offer is a guaranteed 400 on
        # the first request. Rejecting it at load matters because that first
        # request may be a paid eval run mid-flight (Phase 23.5), where the
        # cheapest possible failure is one that happens before any spend.
        from rag_recipes.providers.llm.registry import unsupported_structured_output_modes

        mode = self.llm_structured_output_mode
        if mode is None:
            return self
        # Both roles, because the setting is global but the providers need not
        # be. LLM_PROVIDER=openai + json_schema + JUDGE_LLM_PROVIDER=deepseek
        # would otherwise load clean and die at the first *judge* call — after
        # extraction has already spent the money (Epic 23.3).
        for field, provider in (
            ("llm_provider", self.llm_provider),
            ("judge_llm_provider", self.judge_llm_provider),
        ):
            if provider is None:
                continue
            if mode in unsupported_structured_output_modes(provider):
                raise ValueError(
                    f"llm_structured_output_mode={mode!r} is not supported by {field}={provider!r}"
                )
        return self

    @model_validator(mode="after")
    def _judge_provider_must_not_inherit_a_retargeted_endpoint(self) -> Settings:
        # llm_base_url and llm_provider_label are read by the `openai` registry
        # entry regardless of *who* is asking for it, so with the endpoint
        # retargeted, JUDGE_LLM_PROVIDER=openai builds a judge pointed at the
        # very same third-party endpoint under the very same identity label.
        # The run is then self-judged while looking correctly configured, and
        # the judge cache key cannot detect it either — both sides carry the
        # same label. Since a self-judged comparison is exactly what Epic 23.3
        # exists to prevent, refuse the combination rather than document it.
        if (
            self.llm_base_url
            and self.judge_llm_provider == "openai"
            and self.llm_provider == "openai"
        ):
            raise ValueError(
                "judge_llm_provider='openai' is ambiguous while llm_base_url is set: "
                "the OpenAI transport is shared, so the judge would be built against "
                "the same retargeted endpoint and identity label as the model under "
                "test — a self-judged run that looks correctly configured. Point the "
                "judge at a different registered provider, or clear llm_base_url."
            )
        return self

    @model_validator(mode="after")
    def _base_url_requires_a_provider_label(self) -> Settings:
        # The identity label a built-in registry entry uses is a constant. Setting
        # llm_base_url alone would therefore file another vendor's generations under
        # "openai" — and since the label is part of the extraction cache key, let the
        # two vendors serve each other cached runs. That is the single hazard Epic
        # 23.4 exists to prevent, so it must not be reachable with one env var.
        if self.llm_base_url and not self.llm_provider_label:
            raise ValueError(
                "llm_provider_label is required when llm_base_url is set; the label "
                "is recorded on every ExtractionRun and is part of the extraction "
                "cache key, so a retargeted endpoint must carry its own identity"
            )
        return self

    @model_validator(mode="after")
    def _rerank_provider_supported_when_enabled(self) -> Settings:
        # Only the OpenAI reranker exists in Epic 18.2. Reject an unsupported
        # rerank_provider at load *when reranking is enabled* so chunk text is never
        # silently routed to an unintended/unimplemented vendor (data-egress trust
        # boundary). Disabled config keeps any rerank_provider value (it's inert).
        if self.reranking_enabled and self.rerank_provider != "openai":
            raise ValueError(
                "rerank_provider must be 'openai' when reranking_enabled is true "
                f"(only the OpenAI reranker is supported); got {self.rerank_provider!r}"
            )
        return self

    @model_validator(mode="after")
    def _extraction_recipe_char_band_is_ordered(self) -> Settings:
        # The soft-validation "too short" / "too long" rules require a real band:
        # an inverted or collapsed range (max <= min) would make every recipe
        # both too short and too long. Reject at Settings load so the operator
        # fixes the env rather than getting nonsensical needs_review flags.
        if self.extraction_max_recipe_chars <= self.extraction_min_recipe_chars:
            raise ValueError(
                "extraction_max_recipe_chars must be greater than "
                f"extraction_min_recipe_chars; got "
                f"max={self.extraction_max_recipe_chars}, "
                f"min={self.extraction_min_recipe_chars}"
            )
        return self

    @model_validator(mode="after")
    def _stuck_check_interval_divides_60(self) -> Settings:
        # arq's cron(..., minute={...}) encodes "every N minutes" only when N
        # divides 60 — otherwise the schedule skews at the top of each hour
        # (e.g. N=7 maps to {0,7,14,21,28,35,42,49,56} with a 4-minute gap
        # 56→00). Reject non-divisors at Settings load so the operator picks
        # from the documented valid set rather than discovering skew via a
        # missed sweep tick.
        value = self.stuck_job_check_interval_minutes
        if 60 % value != 0:
            raise ValueError(
                "stuck_job_check_interval_minutes must be a divisor of 60 "
                "(valid: 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60); "
                f"got {value}"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
