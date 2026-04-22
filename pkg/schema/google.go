package schema

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// GoogleTranslator converts between OpenAI chat/embeddings shape and the Google Generative Language (Gemini) API.
// Rerank is not supported.
type GoogleTranslator struct{}

// Name implements Translator.
func (*GoogleTranslator) Name() string { return "google" }

// HealthPath returns the Gemini /v1beta/models listing as a liveness probe.
func (*GoogleTranslator) HealthPath() string { return "/v1beta/models" }

type geminiPart struct {
	Text    string `json:"text,omitempty"`
	Thought bool   `json:"thought,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiGenerationConfig struct {
	Temperature     *float64 `json:"temperature,omitempty"`
	TopP            *float64 `json:"topP,omitempty"`
	MaxOutputTokens *int     `json:"maxOutputTokens,omitempty"`
	StopSequences   []string `json:"stopSequences,omitempty"`
}

type geminiRequest struct {
	Contents          []geminiContent         `json:"contents"`
	SystemInstruction *geminiContent          `json:"systemInstruction,omitempty"`
	GenerationConfig  *geminiGenerationConfig `json:"generationConfig,omitempty"`
}

type geminiResponseCandidate struct {
	Content      geminiContent `json:"content"`
	FinishReason string        `json:"finishReason"`
	Index        int           `json:"index"`
}

type geminiResponse struct {
	Candidates    []geminiResponseCandidate `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
	ModelVersion string `json:"modelVersion"`
}

// ChatBackendRequest translates a canonical chat request to Gemini.
func (*GoogleTranslator) ChatBackendRequest(req *ChatRequest) (*BackendRequest, error) {
	gr := geminiRequest{GenerationConfig: &geminiGenerationConfig{
		Temperature: req.Temperature,
		TopP:        req.TopP,
	}}
	if req.MaxTokens != nil {
		gr.GenerationConfig.MaxOutputTokens = req.MaxTokens
	} else if req.MaxCompletionTok != nil {
		gr.GenerationConfig.MaxOutputTokens = req.MaxCompletionTok
	}
	if stops, err := decodeStringOrArray(req.Stop); err == nil {
		gr.GenerationConfig.StopSequences = stops
	}
	for _, m := range req.Messages {
		text, err := messageContentToText(m.Content)
		if err != nil {
			return nil, err
		}
		switch m.Role {
		case "system":
			if gr.SystemInstruction == nil {
				gr.SystemInstruction = &geminiContent{Parts: []geminiPart{{Text: text}}}
			} else {
				gr.SystemInstruction.Parts = append(gr.SystemInstruction.Parts, geminiPart{Text: text})
			}
		case "user":
			gr.Contents = append(gr.Contents, geminiContent{Role: "user", Parts: []geminiPart{{Text: text}}})
		case "assistant":
			gr.Contents = append(gr.Contents, geminiContent{Role: "model", Parts: []geminiPart{{Text: text}}})
		case "tool":
			gr.Contents = append(gr.Contents, geminiContent{Role: "user", Parts: []geminiPart{{Text: text}}})
		}
	}
	body, err := json.Marshal(gr)
	if err != nil {
		return nil, fmt.Errorf("marshal gemini request: %w", err)
	}
	method := "generateContent"
	path := fmt.Sprintf("/v1beta/models/%s:%s", url.PathEscape(req.Model), method)
	if req.Stream {
		path = fmt.Sprintf("/v1beta/models/%s:streamGenerateContent?alt=sse", url.PathEscape(req.Model))
	}
	return &BackendRequest{
		Method: http.MethodPost,
		Path:   path,
		Body:   body,
	}, nil
}

// ChatResponseFromBytes parses a Gemini non-stream response.
func (*GoogleTranslator) ChatResponseFromBytes(body []byte) (*ChatResponse, error) {
	var g geminiResponse
	if err := json.Unmarshal(body, &g); err != nil {
		return nil, fmt.Errorf("parse gemini response: %w", err)
	}
	resp := &ChatResponse{
		ID:      "gemini-" + strings.TrimSpace(g.ModelVersion),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   g.ModelVersion,
		Usage: &Usage{
			PromptTokens:     g.UsageMetadata.PromptTokenCount,
			CompletionTokens: g.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      g.UsageMetadata.TotalTokenCount,
		},
	}
	for i, c := range g.Candidates {
		var text, thinking strings.Builder
		for _, p := range c.Content.Parts {
			if p.Thought {
				thinking.WriteString(p.Text)
			} else {
				text.WriteString(p.Text)
			}
		}
		contentBytes, _ := json.Marshal(text.String())
		resp.Choices = append(resp.Choices, ChatChoice{
			Index: i,
			Message: Message{
				Role:             "assistant",
				Content:          contentBytes,
				ReasoningContent: thinking.String(),
			},
			FinishReason: geminiFinishReason(c.FinishReason),
		})
	}
	return resp, nil
}

// NewChatStreamReader parses Gemini SSE streaming events.
func (*GoogleTranslator) NewChatStreamReader(body io.ReadCloser) (StreamReader, error) {
	return &googleStreamReader{body: body, sse: NewSSEScanner(body), created: time.Now().Unix()}, nil
}

// EmbeddingsBackendRequest translates to Gemini's :embedContent endpoint.
func (*GoogleTranslator) EmbeddingsBackendRequest(req *EmbeddingsRequest) (*BackendRequest, error) {
	inputs, err := decodeStringOrArray(req.Input)
	if err != nil {
		return nil, fmt.Errorf("embeddings input: %w", err)
	}
	type contentReq struct {
		Content geminiContent `json:"content"`
	}
	type batchReq struct {
		Requests []contentReq `json:"requests"`
	}
	br := batchReq{}
	for _, in := range inputs {
		br.Requests = append(br.Requests, contentReq{
			Content: geminiContent{Parts: []geminiPart{{Text: in}}},
		})
	}
	body, err := json.Marshal(br)
	if err != nil {
		return nil, fmt.Errorf("marshal gemini embeddings request: %w", err)
	}
	return &BackendRequest{
		Method: http.MethodPost,
		Path:   fmt.Sprintf("/v1beta/models/%s:batchEmbedContents", url.PathEscape(req.Model)),
		Body:   body,
	}, nil
}

// EmbeddingsResponseFromBytes parses Gemini batchEmbedContents response.
func (*GoogleTranslator) EmbeddingsResponseFromBytes(body []byte) (*EmbeddingsResponse, error) {
	var g struct {
		Embeddings []struct {
			Values []float64 `json:"values"`
		} `json:"embeddings"`
	}
	if err := json.Unmarshal(body, &g); err != nil {
		return nil, fmt.Errorf("parse gemini embeddings response: %w", err)
	}
	resp := &EmbeddingsResponse{Object: "list"}
	for i, e := range g.Embeddings {
		v, _ := json.Marshal(e.Values)
		resp.Data = append(resp.Data, EmbeddingItem{
			Object:    "embedding",
			Index:     i,
			Embedding: v,
		})
	}
	return resp, nil
}

// RerankBackendRequest returns ErrUnsupportedEndpoint.
func (*GoogleTranslator) RerankBackendRequest(*RerankRequest) (*BackendRequest, error) {
	return nil, ErrUnsupportedEndpoint
}

// RerankResponseFromBytes returns ErrUnsupportedEndpoint.
func (*GoogleTranslator) RerankResponseFromBytes([]byte) (*RerankResponse, error) {
	return nil, ErrUnsupportedEndpoint
}

// ModelsBackendRequest prepares a GET /v1beta/models against the Gemini API.
func (*GoogleTranslator) ModelsBackendRequest() (*BackendRequest, error) {
	return &BackendRequest{Method: http.MethodGet, Path: "/v1beta/models"}, nil
}

// ModelsResponseFromBytes converts the Gemini models list. The Gemini `name` field (e.g., "models/gemini-1.5-pro") is
// stripped of its "models/" prefix to form the canonical id; the full raw entry is preserved under Meta so
// provider-specific fields (version, displayName, inputTokenLimit, supportedGenerationMethods, ...) are not lost.
func (*GoogleTranslator) ModelsResponseFromBytes(body []byte) ([]ModelInfo, error) {
	var wrapper struct {
		Models []json.RawMessage `json:"models"`
	}
	if err := json.Unmarshal(body, &wrapper); err != nil {
		return nil, fmt.Errorf("parse models response: %w", err)
	}
	out := make([]ModelInfo, 0, len(wrapper.Models))
	for _, raw := range wrapper.Models {
		var fields struct {
			Name string `json:"name"`
		}
		if err := json.Unmarshal(raw, &fields); err != nil {
			continue
		}
		id := strings.TrimPrefix(fields.Name, "models/")
		if id == "" {
			continue
		}
		out = append(out, ModelInfo{ID: id, Object: "model", OwnedBy: "google", Meta: raw})
	}
	return out, nil
}

func geminiFinishReason(reason string) string {
	switch reason {
	case "STOP":
		return "stop"
	case "MAX_TOKENS":
		return "length"
	case "SAFETY", "RECITATION", "OTHER":
		return "content_filter"
	default:
		return strings.ToLower(reason)
	}
}

type googleStreamReader struct {
	body             io.ReadCloser
	sse              *SSEScanner
	created          int64
	id               string
	model            string
	seenRole         bool
	promptTokens     int
	completionTokens int
	totalTokens      int
}

func (r *googleStreamReader) Close() error { return r.body.Close() }

func (r *googleStreamReader) Next() ([][]byte, bool, error) {
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
		if bytes.Equal(data, []byte("[DONE]")) {
			return r.finalChunks()
		}
		var g geminiResponse
		if err := json.Unmarshal(data, &g); err != nil {
			return nil, false, fmt.Errorf("parse gemini stream chunk: %w", err)
		}
		if r.model == "" {
			r.model = g.ModelVersion
			r.id = "gemini-" + strings.TrimSpace(g.ModelVersion)
		}
		if g.UsageMetadata.PromptTokenCount > 0 {
			r.promptTokens = g.UsageMetadata.PromptTokenCount
		}
		if g.UsageMetadata.CandidatesTokenCount > 0 {
			r.completionTokens = g.UsageMetadata.CandidatesTokenCount
		}
		if g.UsageMetadata.TotalTokenCount > 0 {
			r.totalTokens = g.UsageMetadata.TotalTokenCount
		}
		var chunks [][]byte
		if !r.seenRole {
			r.seenRole = true
			role := ChatStreamChunk{
				ID: r.id, Object: "chat.completion.chunk", Created: r.created, Model: r.model,
				Choices: []ChatStreamChoice{{Index: 0, Delta: Delta{Role: "assistant"}}},
			}
			b, _ := json.Marshal(role)
			chunks = append(chunks, b)
		}
		for _, c := range g.Candidates {
			for _, p := range c.Content.Parts {
				d := Delta{}
				if p.Thought {
					d.ReasoningContent = p.Text
				} else {
					d.Content = p.Text
				}
				if d.Content == "" && d.ReasoningContent == "" {
					continue
				}
				chunk := ChatStreamChunk{
					ID: r.id, Object: "chat.completion.chunk", Created: r.created, Model: r.model,
					Choices: []ChatStreamChoice{{Index: c.Index, Delta: d}},
				}
				b, _ := json.Marshal(chunk)
				chunks = append(chunks, b)
			}
			if c.FinishReason != "" {
				chunk := ChatStreamChunk{
					ID: r.id, Object: "chat.completion.chunk", Created: r.created, Model: r.model,
					Choices: []ChatStreamChoice{{Index: c.Index, Delta: Delta{}, FinishReason: geminiFinishReason(c.FinishReason)}},
				}
				b, _ := json.Marshal(chunk)
				chunks = append(chunks, b)
			}
		}
		if len(chunks) == 0 {
			continue
		}
		return chunks, false, nil
	}
}

// finalChunks emits a canonical usage-only ChatStreamChunk (choices empty, usage populated) if any token counts were
// captured during the stream, then signals done.
func (r *googleStreamReader) finalChunks() ([][]byte, bool, error) {
	if r.promptTokens == 0 && r.completionTokens == 0 && r.totalTokens == 0 {
		return nil, true, nil
	}
	total := r.totalTokens
	if total == 0 {
		total = r.promptTokens + r.completionTokens
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
			TotalTokens:      total,
		},
	}
	out, err := json.Marshal(chunk)
	if err != nil {
		return nil, false, fmt.Errorf("marshal usage stream chunk: %w", err)
	}
	return [][]byte{out}, true, nil
}
