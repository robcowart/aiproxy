package backend

import (
	"encoding/json"
	"fmt"

	"github.com/robcowart/aiproxy/pkg/schema"
)

// ApplyChatOverrides applies the pool's static parameter overrides to req. Keys whose values map cleanly to fields on
// schema.ChatRequest (temperature, top_p, max_tokens, stream, stop, presence_penalty, frequency_penalty, n, seed) are
// written directly onto req so that translators emit them in their native shape. Remaining keys (top_k, min_p,
// repeat_penalty, logprobs, echo, suffix, plus all llama.cpp-specific keys) are returned in extras for top-level JSON
// merging into the translator-produced backend body via MergeBodyExtras. ApplyChatOverrides assumes the values were
// already type-checked at config load time, so it does best-effort coercion (e.g. accepting an int for a number-typed
// field) and silently ignores any value it cannot coerce — these would have been rejected by config.Validate.
func (p *Pool) ApplyChatOverrides(req *schema.ChatRequest) map[string]any {
	if len(p.Parameters) == 0 {
		return nil
	}
	var extras map[string]any
	for name, val := range p.Parameters {
		switch name {
		case "temperature":
			if f, ok := toFloat(val); ok {
				req.Temperature = &f
			}
		case "top_p":
			if f, ok := toFloat(val); ok {
				req.TopP = &f
			}
		case "max_tokens":
			if n, ok := toInt(val); ok {
				req.MaxTokens = &n
			}
		case "stream":
			if b, ok := val.(bool); ok {
				req.Stream = b
			}
		case "stop":
			if raw, ok := marshalStopValue(val); ok {
				req.Stop = raw
			}
		case "presence_penalty":
			if f, ok := toFloat(val); ok {
				req.PresencePenalty = &f
			}
		case "frequency_penalty":
			if f, ok := toFloat(val); ok {
				req.FrequencyPenalty = &f
			}
		case "n":
			if n, ok := toInt(val); ok {
				req.N = &n
			}
		case "seed":
			if n, ok := toInt64(val); ok {
				req.Seed = &n
			}
		default:
			if extras == nil {
				extras = make(map[string]any, len(p.Parameters))
			}
			extras[name] = val
		}
	}
	return extras
}

// MergeBodyExtras top-level overlays extras onto a JSON object body and re-marshals. The body must encode a JSON
// object; otherwise an error is returned. Returns body unchanged when extras is empty.
func MergeBodyExtras(body []byte, extras map[string]any) ([]byte, error) {
	if len(extras) == 0 {
		return body, nil
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(body, &obj); err != nil {
		return nil, fmt.Errorf("merge body extras: parse body: %w", err)
	}
	if obj == nil {
		obj = make(map[string]json.RawMessage, len(extras))
	}
	for k, v := range extras {
		raw, err := json.Marshal(v)
		if err != nil {
			return nil, fmt.Errorf("merge body extras: marshal %q: %w", k, err)
		}
		obj[k] = raw
	}
	out, err := json.Marshal(obj)
	if err != nil {
		return nil, fmt.Errorf("merge body extras: marshal body: %w", err)
	}
	return out, nil
}

func toFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int32:
		return float64(n), true
	case int64:
		return float64(n), true
	}
	return 0, false
}

func toInt(v any) (int, bool) {
	switch n := v.(type) {
	case int:
		return n, true
	case int32:
		return int(n), true
	case int64:
		return int(n), true
	case float32:
		if float32(int(n)) == n {
			return int(n), true
		}
	case float64:
		if float64(int(n)) == n {
			return int(n), true
		}
	}
	return 0, false
}

func toInt64(v any) (int64, bool) {
	switch n := v.(type) {
	case int:
		return int64(n), true
	case int32:
		return int64(n), true
	case int64:
		return n, true
	case float32:
		if float32(int64(n)) == n {
			return int64(n), true
		}
	case float64:
		if float64(int64(n)) == n {
			return int64(n), true
		}
	}
	return 0, false
}

// marshalStopValue serializes a configured `stop` override (a []any of strings per config validation) to the
// json.RawMessage shape expected by schema.ChatRequest.Stop.
func marshalStopValue(v any) (json.RawMessage, bool) {
	arr, ok := v.([]any)
	if !ok {
		return nil, false
	}
	stops := make([]string, 0, len(arr))
	for _, item := range arr {
		s, ok := item.(string)
		if !ok {
			return nil, false
		}
		stops = append(stops, s)
	}
	raw, err := json.Marshal(stops)
	if err != nil {
		return nil, false
	}
	return raw, true
}
