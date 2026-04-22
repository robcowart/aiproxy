package schema

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// OllamaTranslator converts between canonical OpenAI chat/embeddings shapes and the native Ollama API (/api/chat,
// /api/embed, /api/tags). Streaming uses Ollama's newline-delimited JSON (NDJSON) protocol rather than Server-Sent
// Events (SSE). Rerank is not supported.
type OllamaTranslator struct{}

// Name implements Translator.
func (*OllamaTranslator) Name() string { return "ollama" }

// HealthPath returns Ollama's root endpoint, which cheaply returns the string "Ollama is running".
func (*OllamaTranslator) HealthPath() string { return "/" }

type ollamaMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	Thinking  string           `json:"thinking,omitempty"`
	ToolCalls []ollamaToolCall `json:"tool_calls,omitempty"`
	ToolName  string           `json:"tool_name,omitempty"`
	Images    []string         `json:"images,omitempty"`
}

type ollamaToolCall struct {
	Function ollamaToolCallFunction `json:"function"`
}

type ollamaToolCallFunction struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
}

type ollamaOptions struct {
	Temperature      *float64 `json:"temperature,omitempty"`
	TopP             *float64 `json:"top_p,omitempty"`
	NumPredict       *int     `json:"num_predict,omitempty"`
	Stop             []string `json:"stop,omitempty"`
	Seed             *int64   `json:"seed,omitempty"`
	PresencePenalty  *float64 `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"`
}

type ollamaChatRequest struct {
	Model    string          `json:"model"`
	Messages []ollamaMessage `json:"messages"`
	Stream   bool            `json:"stream"`
	Tools    json.RawMessage `json:"tools,omitempty"`
	Format   json.RawMessage `json:"format,omitempty"`
	Think    *bool           `json:"think,omitempty"`
	Options  *ollamaOptions  `json:"options,omitempty"`
}

type ollamaChatResponse struct {
	Model           string        `json:"model"`
	CreatedAt       string        `json:"created_at"`
	Message         ollamaMessage `json:"message"`
	Done            bool          `json:"done"`
	DoneReason      string        `json:"done_reason"`
	PromptEvalCount int           `json:"prompt_eval_count"`
	EvalCount       int           `json:"eval_count"`
}

// ChatBackendRequest translates a canonical chat request into a native Ollama /api/chat request.
func (*OllamaTranslator) ChatBackendRequest(req *ChatRequest) (*BackendRequest, error) {
	o := ollamaChatRequest{
		Model:  req.Model,
		Stream: req.Stream,
		Tools:  req.Tools,
	}

	opts := ollamaOptions{
		Temperature:      req.Temperature,
		TopP:             req.TopP,
		Seed:             req.Seed,
		PresencePenalty:  req.PresencePenalty,
		FrequencyPenalty: req.FrequencyPenalty,
	}
	if req.MaxTokens != nil {
		opts.NumPredict = req.MaxTokens
	} else if req.MaxCompletionTok != nil {
		opts.NumPredict = req.MaxCompletionTok
	}
	if stops, err := decodeStringOrArray(req.Stop); err == nil {
		opts.Stop = stops
	}
	if !ollamaOptionsEmpty(opts) {
		o.Options = &opts
	}

	if f, ok := ollamaFormatFromResponseFormat(req.ResponseFormat); ok {
		o.Format = f
	}
	if req.ReasoningEffort != "" {
		tr := true
		o.Think = &tr
	}

	for _, m := range req.Messages {
		om, err := ollamaMessageFromCanonical(m)
		if err != nil {
			return nil, err
		}
		o.Messages = append(o.Messages, om)
	}

	body, err := json.Marshal(o)
	if err != nil {
		return nil, fmt.Errorf("marshal ollama chat request: %w", err)
	}
	return &BackendRequest{
		Method: http.MethodPost,
		Path:   "/api/chat",
		Body:   body,
	}, nil
}

// ChatResponseFromBytes parses a native Ollama /api/chat (non-stream) response into canonical form.
func (*OllamaTranslator) ChatResponseFromBytes(body []byte) (*ChatResponse, error) {
	var o ollamaChatResponse
	if err := json.Unmarshal(body, &o); err != nil {
		return nil, fmt.Errorf("parse ollama response: %w", err)
	}
	contentBytes, _ := json.Marshal(o.Message.Content)
	resp := &ChatResponse{
		ID:      ollamaResponseID(o.CreatedAt, o.Model),
		Object:  "chat.completion",
		Created: ollamaCreatedUnix(o.CreatedAt),
		Model:   o.Model,
		Choices: []ChatChoice{{
			Index: 0,
			Message: Message{
				Role:             ollamaMessageRole(o.Message.Role),
				Content:          contentBytes,
				ReasoningContent: o.Message.Thinking,
				ToolCalls:        ollamaToolCallsToOpenAI(o.Message.ToolCalls),
			},
			FinishReason: ollamaFinishReason(o.DoneReason),
		}},
	}
	if o.PromptEvalCount > 0 || o.EvalCount > 0 {
		resp.Usage = &Usage{
			PromptTokens:     o.PromptEvalCount,
			CompletionTokens: o.EvalCount,
			TotalTokens:      o.PromptEvalCount + o.EvalCount,
		}
	}
	return resp, nil
}

// NewChatStreamReader returns an NDJSON stream reader for Ollama /api/chat streaming responses.
func (*OllamaTranslator) NewChatStreamReader(body io.ReadCloser) (StreamReader, error) {
	return &ollamaStreamReader{
		body:    body,
		reader:  bufio.NewReaderSize(body, 64*1024),
		created: time.Now().Unix(),
	}, nil
}

type ollamaEmbeddingsRequest struct {
	Model string          `json:"model"`
	Input json.RawMessage `json:"input"`
}

type ollamaEmbeddingsResponse struct {
	Model           string            `json:"model"`
	Embeddings      []json.RawMessage `json:"embeddings"`
	PromptEvalCount int               `json:"prompt_eval_count"`
}

// EmbeddingsBackendRequest translates to Ollama's /api/embed endpoint.
func (*OllamaTranslator) EmbeddingsBackendRequest(req *EmbeddingsRequest) (*BackendRequest, error) {
	body, err := json.Marshal(ollamaEmbeddingsRequest{Model: req.Model, Input: req.Input})
	if err != nil {
		return nil, fmt.Errorf("marshal ollama embeddings request: %w", err)
	}
	return &BackendRequest{
		Method: http.MethodPost,
		Path:   "/api/embed",
		Body:   body,
	}, nil
}

// EmbeddingsResponseFromBytes parses an Ollama /api/embed response into canonical form.
func (*OllamaTranslator) EmbeddingsResponseFromBytes(body []byte) (*EmbeddingsResponse, error) {
	var o ollamaEmbeddingsResponse
	if err := json.Unmarshal(body, &o); err != nil {
		return nil, fmt.Errorf("parse ollama embeddings response: %w", err)
	}
	resp := &EmbeddingsResponse{Object: "list", Model: o.Model}
	for i, v := range o.Embeddings {
		resp.Data = append(resp.Data, EmbeddingItem{
			Object:    "embedding",
			Index:     i,
			Embedding: v,
		})
	}
	if o.PromptEvalCount > 0 {
		resp.Usage = &Usage{
			PromptTokens: o.PromptEvalCount,
			TotalTokens:  o.PromptEvalCount,
		}
	}
	return resp, nil
}

// RerankBackendRequest returns ErrUnsupportedEndpoint (Ollama has no rerank API).
func (*OllamaTranslator) RerankBackendRequest(*RerankRequest) (*BackendRequest, error) {
	return nil, ErrUnsupportedEndpoint
}

// RerankResponseFromBytes returns ErrUnsupportedEndpoint.
func (*OllamaTranslator) RerankResponseFromBytes([]byte) (*RerankResponse, error) {
	return nil, ErrUnsupportedEndpoint
}

// ModelsBackendRequest prepares a GET /api/tags against the Ollama API.
func (*OllamaTranslator) ModelsBackendRequest() (*BackendRequest, error) {
	return &BackendRequest{Method: http.MethodGet, Path: "/api/tags"}, nil
}

// ModelsResponseFromBytes converts Ollama's /api/tags listing to canonical ModelInfo entries. The `name` field becomes
// the canonical id; `modified_at` (RFC3339) becomes Unix seconds; the full raw entry is preserved under Meta so
// provider-specific fields (size, digest, details, ...) are not lost.
func (*OllamaTranslator) ModelsResponseFromBytes(body []byte) ([]ModelInfo, error) {
	var wrapper struct {
		Models []json.RawMessage `json:"models"`
	}
	if err := json.Unmarshal(body, &wrapper); err != nil {
		return nil, fmt.Errorf("parse models response: %w", err)
	}
	out := make([]ModelInfo, 0, len(wrapper.Models))
	for _, raw := range wrapper.Models {
		var fields struct {
			Name       string `json:"name"`
			ModifiedAt string `json:"modified_at"`
		}
		if err := json.Unmarshal(raw, &fields); err != nil {
			continue
		}
		if fields.Name == "" {
			continue
		}
		mi := ModelInfo{ID: fields.Name, Object: "model", OwnedBy: "ollama", Meta: raw}
		if fields.ModifiedAt != "" {
			if t, err := time.Parse(time.RFC3339, fields.ModifiedAt); err == nil {
				mi.Created = t.Unix()
			}
		}
		out = append(out, mi)
	}
	return out, nil
}

// ollamaStreamReader reads NDJSON lines from the backend and emits canonical ChatStreamChunk payloads.
type ollamaStreamReader struct {
	body             io.ReadCloser
	reader           *bufio.Reader
	created          int64
	id               string
	model            string
	seenRole         bool
	promptTokens     int
	completionTokens int
}

func (r *ollamaStreamReader) Close() error { return r.body.Close() }

func (r *ollamaStreamReader) Next() ([][]byte, bool, error) {
	for {
		line, err := r.readLine()
		if err == io.EOF {
			return r.finalChunks()
		}
		if err != nil {
			return nil, false, err
		}
		trimmed := bytes.TrimSpace(line)
		if len(trimmed) == 0 {
			continue
		}
		var o ollamaChatResponse
		if err := json.Unmarshal(trimmed, &o); err != nil {
			return nil, false, fmt.Errorf("parse ollama stream chunk: %w", err)
		}
		if r.model == "" && o.Model != "" {
			r.model = o.Model
			r.id = ollamaResponseID(o.CreatedAt, o.Model)
		}
		if o.PromptEvalCount > 0 {
			r.promptTokens = o.PromptEvalCount
		}
		if o.EvalCount > 0 {
			r.completionTokens = o.EvalCount
		}

		var chunks [][]byte
		if !r.seenRole {
			r.seenRole = true
			role := r.baseChunk(Delta{Role: "assistant"}, "")
			b, err := json.Marshal(role)
			if err != nil {
				return nil, false, fmt.Errorf("marshal ollama stream chunk: %w", err)
			}
			chunks = append(chunks, b)
		}

		delta := Delta{
			Content:          o.Message.Content,
			ReasoningContent: o.Message.Thinking,
			ToolCalls:        ollamaToolCallsToOpenAI(o.Message.ToolCalls),
		}
		if delta.Content != "" || delta.ReasoningContent != "" || len(delta.ToolCalls) > 0 {
			chunk := r.baseChunk(delta, "")
			b, err := json.Marshal(chunk)
			if err != nil {
				return nil, false, fmt.Errorf("marshal ollama stream chunk: %w", err)
			}
			chunks = append(chunks, b)
		}
		if o.Done {
			finish := ollamaFinishReason(o.DoneReason)
			if finish == "" {
				finish = "stop"
			}
			finishChunk := r.baseChunk(Delta{}, finish)
			b, err := json.Marshal(finishChunk)
			if err != nil {
				return nil, false, fmt.Errorf("marshal ollama stream chunk: %w", err)
			}
			chunks = append(chunks, b)
			usageChunks, _, err := r.finalChunks()
			if err != nil {
				return nil, false, err
			}
			chunks = append(chunks, usageChunks...)
			return chunks, true, nil
		}
		if len(chunks) == 0 {
			continue
		}
		return chunks, false, nil
	}
}

// readLine returns the next newline-terminated line from the backend. A trailing unterminated line is surfaced as a
// successful read; the next call will then return io.EOF.
func (r *ollamaStreamReader) readLine() ([]byte, error) {
	line, err := r.reader.ReadBytes('\n')
	if len(line) > 0 {
		return line, nil
	}
	return nil, err
}

// finalChunks emits a trailing usage-only ChatStreamChunk when token counts were observed during the stream. Mirrors
// the pattern used by the Anthropic and Google stream readers.
func (r *ollamaStreamReader) finalChunks() ([][]byte, bool, error) {
	if r.promptTokens == 0 && r.completionTokens == 0 {
		return nil, true, nil
	}
	chunk := ChatStreamChunk{
		ID:      r.id,
		Object:  "chat.completion.chunk",
		Created: r.created,
		Model:   r.model,
		Choices: []ChatStreamChoice{},
		Usage: &Usage{
			PromptTokens:     r.promptTokens,
			CompletionTokens: r.completionTokens,
			TotalTokens:      r.promptTokens + r.completionTokens,
		},
	}
	out, err := json.Marshal(chunk)
	if err != nil {
		return nil, false, fmt.Errorf("marshal usage stream chunk: %w", err)
	}
	return [][]byte{out}, true, nil
}

func (r *ollamaStreamReader) baseChunk(d Delta, finish string) ChatStreamChunk {
	return ChatStreamChunk{
		ID:      r.id,
		Object:  "chat.completion.chunk",
		Created: r.created,
		Model:   r.model,
		Choices: []ChatStreamChoice{{
			Index:        0,
			Delta:        d,
			FinishReason: finish,
		}},
	}
}

// ollamaMessageFromCanonical translates a canonical Message to Ollama's native shape. Tool-call arguments in OpenAI
// format are JSON strings; Ollama expects a JSON object, so we re-parse them best-effort.
func ollamaMessageFromCanonical(m Message) (ollamaMessage, error) {
	text, err := messageContentToText(m.Content)
	if err != nil {
		return ollamaMessage{}, err
	}
	om := ollamaMessage{
		Role:     m.Role,
		Content:  text,
		Thinking: m.ReasoningContent,
		ToolName: m.Name,
	}
	if m.Role == "tool" && m.ToolCallID != "" && om.ToolName == "" {
		om.ToolName = m.ToolCallID
	}
	if len(m.ToolCalls) > 0 {
		tcs, err := openAIToolCallsToOllama(m.ToolCalls)
		if err != nil {
			return ollamaMessage{}, err
		}
		om.ToolCalls = tcs
	}
	return om, nil
}

// openAIToolCallsToOllama parses an OpenAI-shaped tool_calls array and converts each call's string arguments to a JSON
// object as Ollama expects.
func openAIToolCallsToOllama(raw json.RawMessage) ([]ollamaToolCall, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var calls []struct {
		Function struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		} `json:"function"`
	}
	if err := json.Unmarshal(raw, &calls); err != nil {
		return nil, fmt.Errorf("decode tool_calls: %w", err)
	}
	out := make([]ollamaToolCall, 0, len(calls))
	for _, c := range calls {
		args := c.Function.Arguments
		var asString string
		if err := json.Unmarshal(args, &asString); err == nil {
			if strings.TrimSpace(asString) == "" {
				args = nil
			} else {
				args = json.RawMessage(asString)
			}
		}
		out = append(out, ollamaToolCall{
			Function: ollamaToolCallFunction{Name: c.Function.Name, Arguments: args},
		})
	}
	return out, nil
}

// ollamaToolCallsToOpenAI re-serializes Ollama-shaped tool_calls to the OpenAI shape, encoding each function's
// arguments object as a JSON string. Returns nil when there are no tool calls.
func ollamaToolCallsToOpenAI(calls []ollamaToolCall) json.RawMessage {
	if len(calls) == 0 {
		return nil
	}
	type openAIFunction struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	}
	type openAIToolCall struct {
		ID       string         `json:"id,omitempty"`
		Type     string         `json:"type"`
		Function openAIFunction `json:"function"`
	}
	out := make([]openAIToolCall, 0, len(calls))
	for _, c := range calls {
		args := ""
		if len(c.Function.Arguments) > 0 {
			args = string(c.Function.Arguments)
		}
		out = append(out, openAIToolCall{
			Type:     "function",
			Function: openAIFunction{Name: c.Function.Name, Arguments: args},
		})
	}
	b, err := json.Marshal(out)
	if err != nil {
		return nil
	}
	return b
}

// ollamaFormatFromResponseFormat maps OpenAI response_format to Ollama's `format` field:
//   - {"type":"json_object"}  -> "json"
//   - {"type":"json_schema","json_schema":{"schema":<obj>}} -> <obj>
//   - {"type":"text"} or unset -> none
func ollamaFormatFromResponseFormat(raw json.RawMessage) (json.RawMessage, bool) {
	if len(raw) == 0 {
		return nil, false
	}
	var rf struct {
		Type       string `json:"type"`
		JSONSchema struct {
			Schema json.RawMessage `json:"schema"`
		} `json:"json_schema"`
	}
	if err := json.Unmarshal(raw, &rf); err != nil {
		return nil, false
	}
	switch rf.Type {
	case "json_object":
		return json.RawMessage(`"json"`), true
	case "json_schema":
		if len(rf.JSONSchema.Schema) > 0 {
			return rf.JSONSchema.Schema, true
		}
	}
	return nil, false
}

// ollamaOptionsEmpty reports whether every field in opts is the zero value; used to omit `options` entirely when no
// knobs were set.
func ollamaOptionsEmpty(o ollamaOptions) bool {
	return o.Temperature == nil && o.TopP == nil && o.NumPredict == nil && len(o.Stop) == 0 &&
		o.Seed == nil && o.PresencePenalty == nil && o.FrequencyPenalty == nil
}

// ollamaFinishReason maps Ollama's done_reason to the OpenAI finish_reason vocabulary.
func ollamaFinishReason(reason string) string {
	switch reason {
	case "stop", "":
		return "stop"
	case "length":
		return "length"
	case "load":
		return "stop"
	default:
		return reason
	}
}

// ollamaMessageRole normalizes Ollama's role to an OpenAI-compatible role, defaulting to "assistant" when empty (as
// returned by /api/chat response messages).
func ollamaMessageRole(r string) string {
	if r == "" {
		return "assistant"
	}
	return r
}

// ollamaCreatedUnix parses Ollama's RFC3339 created_at timestamp, falling back to the current time when parsing fails
// or the field is empty.
func ollamaCreatedUnix(s string) int64 {
	if s == "" {
		return time.Now().Unix()
	}
	if t, err := time.Parse(time.RFC3339Nano, s); err == nil {
		return t.Unix()
	}
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t.Unix()
	}
	return time.Now().Unix()
}

// ollamaResponseID synthesizes a stable id from the model and created_at timestamp. Ollama does not emit a response
// id, so we fabricate one that round-trips usefully in logs.
func ollamaResponseID(createdAt, model string) string {
	ts := strings.TrimSpace(createdAt)
	if ts == "" {
		ts = fmt.Sprintf("%d", time.Now().Unix())
	}
	m := strings.TrimSpace(model)
	if m == "" {
		return "ollama-" + ts
	}
	return "ollama-" + m + "-" + ts
}
