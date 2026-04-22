package schema

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// AnthropicTranslator converts between OpenAI chat shape and Anthropic Messages API (/v1/messages). Embeddings and
// rerank are not supported.
type AnthropicTranslator struct{}

// Name implements Translator.
func (*AnthropicTranslator) Name() string { return "anthropic" }

// HealthPath returns the Anthropic /v1/models listing (used as a liveness probe).
func (*AnthropicTranslator) HealthPath() string { return "/v1/models" }

type anthropicMessage struct {
	Role    string          `json:"role"`
	Content []anthropicPart `json:"content"`
}

type anthropicPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	Thinking string `json:"thinking,omitempty"`
}

type anthropicRequest struct {
	Model       string             `json:"model"`
	Messages    []anthropicMessage `json:"messages"`
	System      string             `json:"system,omitempty"`
	MaxTokens   int                `json:"max_tokens"`
	Temperature *float64           `json:"temperature,omitempty"`
	TopP        *float64           `json:"top_p,omitempty"`
	Stream      bool               `json:"stream,omitempty"`
	StopSeq     []string           `json:"stop_sequences,omitempty"`
}

type anthropicResponse struct {
	ID         string          `json:"id"`
	Model      string          `json:"model"`
	Role       string          `json:"role"`
	Content    []anthropicPart `json:"content"`
	StopReason string          `json:"stop_reason"`
	Usage      struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

// ChatBackendRequest translates a canonical chat request to Anthropic Messages.
func (*AnthropicTranslator) ChatBackendRequest(req *ChatRequest) (*BackendRequest, error) {
	a := anthropicRequest{
		Model:       req.Model,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
	}
	if req.MaxTokens != nil {
		a.MaxTokens = *req.MaxTokens
	} else if req.MaxCompletionTok != nil {
		a.MaxTokens = *req.MaxCompletionTok
	} else {
		a.MaxTokens = 1024
	}
	if stops, err := decodeStringOrArray(req.Stop); err == nil {
		a.StopSeq = stops
	}
	for _, m := range req.Messages {
		text, err := messageContentToText(m.Content)
		if err != nil {
			return nil, err
		}
		switch m.Role {
		case "system":
			if a.System != "" {
				a.System += "\n\n"
			}
			a.System += text
		case "user", "assistant":
			a.Messages = append(a.Messages, anthropicMessage{
				Role:    m.Role,
				Content: []anthropicPart{{Type: "text", Text: text}},
			})
		case "tool":
			a.Messages = append(a.Messages, anthropicMessage{
				Role:    "user",
				Content: []anthropicPart{{Type: "text", Text: text}},
			})
		}
	}
	body, err := json.Marshal(a)
	if err != nil {
		return nil, fmt.Errorf("marshal anthropic request: %w", err)
	}
	hdr := http.Header{}
	hdr.Set("anthropic-version", "2023-06-01")
	return &BackendRequest{
		Method:  http.MethodPost,
		Path:    "/v1/messages",
		Body:    body,
		Headers: hdr,
	}, nil
}

// ChatResponseFromBytes parses an Anthropic Messages response into canonical form.
func (*AnthropicTranslator) ChatResponseFromBytes(body []byte) (*ChatResponse, error) {
	var a anthropicResponse
	if err := json.Unmarshal(body, &a); err != nil {
		return nil, fmt.Errorf("parse anthropic response: %w", err)
	}
	var text, thinking strings.Builder
	for _, p := range a.Content {
		switch p.Type {
		case "text":
			text.WriteString(p.Text)
		case "thinking":
			thinking.WriteString(p.Thinking)
		}
	}
	contentBytes, _ := json.Marshal(text.String())
	return &ChatResponse{
		ID:      a.ID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   a.Model,
		Choices: []ChatChoice{{
			Index: 0,
			Message: Message{
				Role:             "assistant",
				Content:          contentBytes,
				ReasoningContent: thinking.String(),
			},
			FinishReason: anthropicFinishReason(a.StopReason),
		}},
		Usage: &Usage{
			PromptTokens:     a.Usage.InputTokens,
			CompletionTokens: a.Usage.OutputTokens,
			TotalTokens:      a.Usage.InputTokens + a.Usage.OutputTokens,
		},
	}, nil
}

// NewChatStreamReader parses Anthropic SSE events and emits canonical chunks.
func (*AnthropicTranslator) NewChatStreamReader(body io.ReadCloser) (StreamReader, error) {
	return &anthropicStreamReader{body: body, sse: NewSSEScanner(body), created: time.Now().Unix()}, nil
}

// EmbeddingsBackendRequest returns ErrUnsupportedEndpoint (Anthropic has no embeddings endpoint).
func (*AnthropicTranslator) EmbeddingsBackendRequest(*EmbeddingsRequest) (*BackendRequest, error) {
	return nil, ErrUnsupportedEndpoint
}

// EmbeddingsResponseFromBytes returns ErrUnsupportedEndpoint.
func (*AnthropicTranslator) EmbeddingsResponseFromBytes([]byte) (*EmbeddingsResponse, error) {
	return nil, ErrUnsupportedEndpoint
}

// RerankBackendRequest returns ErrUnsupportedEndpoint.
func (*AnthropicTranslator) RerankBackendRequest(*RerankRequest) (*BackendRequest, error) {
	return nil, ErrUnsupportedEndpoint
}

// RerankResponseFromBytes returns ErrUnsupportedEndpoint.
func (*AnthropicTranslator) RerankResponseFromBytes([]byte) (*RerankResponse, error) {
	return nil, ErrUnsupportedEndpoint
}

// ModelsBackendRequest prepares a GET /v1/models against the Anthropic API.
func (*AnthropicTranslator) ModelsBackendRequest() (*BackendRequest, error) {
	return &BackendRequest{Method: http.MethodGet, Path: "/v1/models"}, nil
}

// ModelsResponseFromBytes converts the Anthropic models list to canonical ModelInfo entries. Anthropic returns
// `created_at` as an RFC3339 timestamp; it is converted to Unix seconds. `display_name` is surfaced under `meta`.
func (*AnthropicTranslator) ModelsResponseFromBytes(body []byte) ([]ModelInfo, error) {
	var wrapper struct {
		Data []struct {
			ID          string `json:"id"`
			Type        string `json:"type"`
			CreatedAt   string `json:"created_at"`
			DisplayName string `json:"display_name"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &wrapper); err != nil {
		return nil, fmt.Errorf("parse models response: %w", err)
	}
	out := make([]ModelInfo, 0, len(wrapper.Data))
	for _, d := range wrapper.Data {
		mi := ModelInfo{ID: d.ID, Object: "model", OwnedBy: "anthropic"}
		if d.CreatedAt != "" {
			if t, err := time.Parse(time.RFC3339, d.CreatedAt); err == nil {
				mi.Created = t.Unix()
			}
		}
		if d.DisplayName != "" {
			if meta, err := json.Marshal(map[string]string{"display_name": d.DisplayName}); err == nil {
				mi.Meta = meta
			}
		}
		out = append(out, mi)
	}
	return out, nil
}

func anthropicFinishReason(reason string) string {
	switch reason {
	case "end_turn", "stop_sequence":
		return "stop"
	case "max_tokens":
		return "length"
	case "tool_use":
		return "tool_calls"
	default:
		return reason
	}
}

// decodeStringOrArray unmarshals raw as either a JSON string or a JSON array of strings, returning a normalized
// []string. An empty/null raw yields (nil, nil). A malformed value that matches neither shape yields an error.
func decodeStringOrArray(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var single string
	if err := json.Unmarshal(raw, &single); err == nil {
		return []string{single}, nil
	}
	var arr []string
	if err := json.Unmarshal(raw, &arr); err != nil {
		return nil, fmt.Errorf("expected string or []string: %w", err)
	}
	return arr, nil
}

// messageContentToText extracts plain text from the OpenAI content union (either a JSON string or an array of {type,
// text} parts).
func messageContentToText(raw json.RawMessage) (string, error) {
	if len(raw) == 0 {
		return "", nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s, nil
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err != nil {
		return "", fmt.Errorf("decode message content: %w", err)
	}
	var b strings.Builder
	for _, p := range parts {
		if p.Type == "text" {
			b.WriteString(p.Text)
		}
	}
	return b.String(), nil
}

type anthropicStreamReader struct {
	body         io.ReadCloser
	sse          *SSEScanner
	id           string
	model        string
	created      int64
	idxMap       map[int]string
	inputTokens  int
	outputTokens int
}

func (r *anthropicStreamReader) Close() error { return r.body.Close() }

func (r *anthropicStreamReader) Next() ([][]byte, bool, error) {
	if r.idxMap == nil {
		r.idxMap = make(map[int]string)
	}
	for {
		ev, err := r.sse.Next()
		if err == io.EOF {
			return r.finalChunks()
		}
		if err != nil {
			return nil, false, err
		}
		if ev == nil || len(ev.Data) == 0 {
			continue
		}
		data := bytes.TrimSpace(ev.Data)
		chunk, emit, done, err := r.handleEvent(ev.Event, data)
		if err != nil {
			return nil, false, err
		}
		if done {
			return r.finalChunks()
		}
		if !emit {
			continue
		}
		out, err := json.Marshal(chunk)
		if err != nil {
			return nil, false, fmt.Errorf("marshal chat stream chunk: %w", err)
		}
		return [][]byte{out}, false, nil
	}
}

// finalChunks emits a canonical usage-only ChatStreamChunk (choices empty, usage populated) if any token counts were
// captured during the stream, then signals done. When no usage was observed (unusual for Anthropic), it just signals
// done with no chunks.
func (r *anthropicStreamReader) finalChunks() ([][]byte, bool, error) {
	if r.inputTokens == 0 && r.outputTokens == 0 {
		return nil, true, nil
	}
	chunk := ChatStreamChunk{
		ID:      r.id,
		Object:  "chat.completion.chunk",
		Created: r.created,
		Model:   r.model,
		Choices: []ChatStreamChoice{},
		Usage: &Usage{
			PromptTokens:     r.inputTokens,
			CompletionTokens: r.outputTokens,
			TotalTokens:      r.inputTokens + r.outputTokens,
		},
	}
	out, err := json.Marshal(chunk)
	if err != nil {
		return nil, false, fmt.Errorf("marshal usage stream chunk: %w", err)
	}
	return [][]byte{out}, true, nil
}

func (r *anthropicStreamReader) handleEvent(event string, data []byte) (ChatStreamChunk, bool, bool, error) {
	switch event {
	case "message_start":
		var m struct {
			Message struct {
				ID    string `json:"id"`
				Model string `json:"model"`
				Usage struct {
					InputTokens  int `json:"input_tokens"`
					OutputTokens int `json:"output_tokens"`
				} `json:"usage"`
			} `json:"message"`
		}
		if err := json.Unmarshal(data, &m); err != nil {
			return ChatStreamChunk{}, false, false, err
		}
		r.id = m.Message.ID
		r.model = m.Message.Model
		if m.Message.Usage.InputTokens > 0 {
			r.inputTokens = m.Message.Usage.InputTokens
		}
		if m.Message.Usage.OutputTokens > 0 {
			r.outputTokens = m.Message.Usage.OutputTokens
		}
		return r.baseChunk(Delta{Role: "assistant"}, ""), true, false, nil

	case "content_block_start":
		var b struct {
			Index        int `json:"index"`
			ContentBlock struct {
				Type string `json:"type"`
			} `json:"content_block"`
		}
		if err := json.Unmarshal(data, &b); err != nil {
			return ChatStreamChunk{}, false, false, err
		}
		r.idxMap[b.Index] = b.ContentBlock.Type
		return ChatStreamChunk{}, false, false, nil

	case "content_block_delta":
		var b struct {
			Index int `json:"index"`
			Delta struct {
				Type     string `json:"type"`
				Text     string `json:"text"`
				Thinking string `json:"thinking"`
			} `json:"delta"`
		}
		if err := json.Unmarshal(data, &b); err != nil {
			return ChatStreamChunk{}, false, false, err
		}
		delta := Delta{}
		switch b.Delta.Type {
		case "text_delta":
			delta.Content = b.Delta.Text
		case "thinking_delta":
			delta.ReasoningContent = b.Delta.Thinking
		default:
			return ChatStreamChunk{}, false, false, nil
		}
		return r.baseChunk(delta, ""), true, false, nil

	case "message_delta":
		var m struct {
			Delta struct {
				StopReason string `json:"stop_reason"`
			} `json:"delta"`
			Usage struct {
				InputTokens  int `json:"input_tokens"`
				OutputTokens int `json:"output_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal(data, &m); err != nil {
			return ChatStreamChunk{}, false, false, err
		}
		if m.Usage.InputTokens > 0 {
			r.inputTokens = m.Usage.InputTokens
		}
		if m.Usage.OutputTokens > 0 {
			r.outputTokens = m.Usage.OutputTokens
		}
		if m.Delta.StopReason == "" {
			return ChatStreamChunk{}, false, false, nil
		}
		return r.baseChunk(Delta{}, anthropicFinishReason(m.Delta.StopReason)), true, false, nil

	case "message_stop":
		return ChatStreamChunk{}, false, true, nil

	default:
		return ChatStreamChunk{}, false, false, nil
	}
}

func (r *anthropicStreamReader) baseChunk(d Delta, finish string) ChatStreamChunk {
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
