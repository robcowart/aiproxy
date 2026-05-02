// Package schema defines the canonical OpenAI-shaped request/response types used within aiproxy and the Translator
// interface each provider implements to convert to/from its native API.
package schema

import (
	"encoding/json"
	"fmt"
)

// NormalizeStreamOptions enforces include_usage=true on the stream_options sub-object sent to backends, preserving any
// other fields the client set. It returns the possibly-rewritten raw JSON plus clientOptedOut, which is true iff the
// caller's original raw explicitly set include_usage to false. Callers use clientOptedOut to decide whether to strip
// the usage-only chunk from the client-visible SSE stream.
//
// Behavior:
//   - raw is empty/null: returns {"include_usage":true}, false, nil.
//   - raw is {}: returns {"include_usage":true}, false, nil.
//   - raw already has include_usage:true: returns raw unchanged, false, nil.
//   - raw has include_usage:false: returns raw with include_usage:true, true, nil.
//   - raw is not a JSON object: returns nil, false, error.
func NormalizeStreamOptions(raw json.RawMessage) (json.RawMessage, bool, error) {
	if len(raw) == 0 || string(raw) == "null" {
		out, err := json.Marshal(map[string]any{"include_usage": true})
		if err != nil {
			return nil, false, err
		}
		return out, false, nil
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return nil, false, fmt.Errorf("stream_options: %w", err)
	}
	if fields == nil {
		fields = map[string]json.RawMessage{}
	}
	clientOptedOut := false
	if iu, ok := fields["include_usage"]; ok {
		var v bool
		if err := json.Unmarshal(iu, &v); err != nil {
			return nil, false, fmt.Errorf("stream_options.include_usage: %w", err)
		}
		if !v {
			clientOptedOut = true
		}
	}
	trueRaw, err := json.Marshal(true)
	if err != nil {
		return nil, false, err
	}
	fields["include_usage"] = trueRaw
	out, err := json.Marshal(fields)
	if err != nil {
		return nil, false, err
	}
	return out, clientOptedOut, nil
}

// ChatRequest is the canonical OpenAI /v1/chat/completions request body. Content fields are kept as json.RawMessage so
// that string and multi-part content shapes both round-trip without loss.
type ChatRequest struct {
	Model            string          `json:"model"`
	Messages         []Message       `json:"messages"`
	Stream           bool            `json:"stream,omitempty"`
	Temperature      *float64        `json:"temperature,omitempty"`
	TopP             *float64        `json:"top_p,omitempty"`
	MaxTokens        *int            `json:"max_tokens,omitempty"`
	MaxCompletionTok *int            `json:"max_completion_tokens,omitempty"`
	N                *int            `json:"n,omitempty"`
	Stop             json.RawMessage `json:"stop,omitempty"`
	PresencePenalty  *float64        `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64        `json:"frequency_penalty,omitempty"`
	Seed             *int64          `json:"seed,omitempty"`
	User             string          `json:"user,omitempty"`
	Tools            json.RawMessage `json:"tools,omitempty"`
	ToolChoice       json.RawMessage `json:"tool_choice,omitempty"`
	ResponseFormat   json.RawMessage `json:"response_format,omitempty"`
	ReasoningEffort  string          `json:"reasoning_effort,omitempty"`
	StreamOptions    json.RawMessage `json:"stream_options,omitempty"`
}

// Message is a chat message (prompt or response).
type Message struct {
	Role             string          `json:"role"`
	Content          json.RawMessage `json:"content,omitempty"`
	ReasoningContent string          `json:"reasoning_content,omitempty"`
	Name             string          `json:"name,omitempty"`
	ToolCalls        json.RawMessage `json:"tool_calls,omitempty"`
	ToolCallID       string          `json:"tool_call_id,omitempty"`
}

// ChatResponse is the canonical OpenAI /v1/chat/completions non-stream body. Timings is an opaque pass-through for
// backend-specific timing blocks (e.g., llama.cpp's top-level `timings` object).
type ChatResponse struct {
	ID                string          `json:"id"`
	Object            string          `json:"object"`
	Created           int64           `json:"created"`
	Model             string          `json:"model"`
	SystemFingerprint string          `json:"system_fingerprint,omitempty"`
	Choices           []ChatChoice    `json:"choices"`
	Usage             *Usage          `json:"usage,omitempty"`
	Timings           json.RawMessage `json:"timings,omitempty"`
}

// ChatChoice is a single completion choice.
type ChatChoice struct {
	Index        int             `json:"index"`
	Message      Message         `json:"message"`
	FinishReason string          `json:"finish_reason"`
	Logprobs     json.RawMessage `json:"logprobs,omitempty"`
}

// ChatStreamChunk is the canonical streamed delta payload. Timings is an opaque pass-through for backend-specific
// timing blocks that some providers (e.g., llama.cpp) append to the final chunk.
type ChatStreamChunk struct {
	ID                string             `json:"id"`
	Object            string             `json:"object"`
	Created           int64              `json:"created"`
	Model             string             `json:"model"`
	SystemFingerprint string             `json:"system_fingerprint,omitempty"`
	Choices           []ChatStreamChoice `json:"choices"`
	Usage             *Usage             `json:"usage,omitempty"`
	Timings           json.RawMessage    `json:"timings,omitempty"`
}

// ChatStreamChoice is one choice within a streamed chunk.
type ChatStreamChoice struct {
	Index        int             `json:"index"`
	Delta        Delta           `json:"delta"`
	FinishReason string          `json:"finish_reason,omitempty"`
	Logprobs     json.RawMessage `json:"logprobs,omitempty"`
}

// Delta carries a streamed partial message.
type Delta struct {
	Role             string          `json:"role,omitempty"`
	Content          string          `json:"content,omitempty"`
	ReasoningContent string          `json:"reasoning_content,omitempty"`
	ToolCalls        json.RawMessage `json:"tool_calls,omitempty"`
}

// Usage reports token accounting. The *TokensDetails fields are kept as json.RawMessage so that backend-specific
// breakdowns (e.g., OpenAI's cached_tokens/reasoning_tokens, llama.cpp's cached_tokens) round-trip without loss.
type Usage struct {
	PromptTokens            int             `json:"prompt_tokens"`
	CompletionTokens        int             `json:"completion_tokens"`
	TotalTokens             int             `json:"total_tokens"`
	PromptTokensDetails     json.RawMessage `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails json.RawMessage `json:"completion_tokens_details,omitempty"`
}

// ModelInfo describes a single model exposed by a backend in the /v1/models listing. Meta is an opaque JSON
// pass-through for provider-specific extras (e.g., llama.cpp's `meta` object with vocab/context/embedding/param
// counts) so no fields are silently dropped. Aliases captures alternate identifiers that some backends (notably
// llama.cpp) advertise alongside `id`, so the frontend can match a pool's configured model name even when the backend
// reports a quantization-suffixed canonical id.
type ModelInfo struct {
	ID      string          `json:"id"`
	Aliases []string        `json:"aliases,omitempty"`
	Object  string          `json:"object,omitempty"`
	Created int64           `json:"created,omitempty"`
	OwnedBy string          `json:"owned_by,omitempty"`
	Meta    json.RawMessage `json:"meta,omitempty"`
}

// EmbeddingsRequest is the canonical /v1/embeddings request body.
type EmbeddingsRequest struct {
	Model          string          `json:"model"`
	Input          json.RawMessage `json:"input"`
	EncodingFormat string          `json:"encoding_format,omitempty"`
	Dimensions     *int            `json:"dimensions,omitempty"`
	User           string          `json:"user,omitempty"`
}

// EmbeddingsResponse is the canonical /v1/embeddings response body.
type EmbeddingsResponse struct {
	Object string          `json:"object"`
	Data   []EmbeddingItem `json:"data"`
	Model  string          `json:"model"`
	Usage  *Usage          `json:"usage,omitempty"`
}

// EmbeddingItem is a single embedding vector in the response data array.
type EmbeddingItem struct {
	Object    string          `json:"object"`
	Index     int             `json:"index"`
	Embedding json.RawMessage `json:"embedding"`
}

// RerankRequest is the canonical rerank request body (llama.cpp/Cohere-style).
type RerankRequest struct {
	Model           string   `json:"model"`
	Query           string   `json:"query"`
	Documents       []string `json:"documents"`
	TopN            *int     `json:"top_n,omitempty"`
	ReturnDocuments *bool    `json:"return_documents,omitempty"`
}

// RerankResponse is the canonical rerank response body.
type RerankResponse struct {
	Object  string       `json:"object"`
	Model   string       `json:"model,omitempty"`
	Results []RerankItem `json:"results"`
	Usage   *Usage       `json:"usage,omitempty"`
}

// RerankItem is a single rerank result item.
type RerankItem struct {
	Index          int             `json:"index"`
	RelevanceScore float64         `json:"relevance_score"`
	Document       json.RawMessage `json:"document,omitempty"`
}
