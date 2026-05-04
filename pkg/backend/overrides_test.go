package backend

import (
	"encoding/json"
	"testing"

	"github.com/robcowart/aiproxy/pkg/schema"
	"github.com/stretchr/testify/assert"
)

func TestApplyChatOverrides_CanonicalAndExtras(t *testing.T) {
	pool := &Pool{
		Parameters: map[string]any{
			"temperature": 0.25,
			"stream":      false,
			"max_tokens":  256,
			"stop":        []any{"</s>", "User:"},
			"top_k":       40,
			"mirostat":    2,
			"grammar":     "root ::= .",
		},
	}
	req := &schema.ChatRequest{Stream: true}
	extras := pool.ApplyChatOverrides(req)

	if assert.NotNil(t, req.Temperature) {
		assert.Equal(t, 0.25, *req.Temperature)
	}
	assert.False(t, req.Stream, "stream override should flip request to non-streaming")
	if assert.NotNil(t, req.MaxTokens) {
		assert.Equal(t, 256, *req.MaxTokens)
	}
	assert.JSONEq(t, `["</s>","User:"]`, string(req.Stop))

	assert.Contains(t, extras, "top_k")
	assert.Contains(t, extras, "mirostat")
	assert.Contains(t, extras, "grammar")
	assert.NotContains(t, extras, "temperature")
	assert.NotContains(t, extras, "stream")
}

func TestApplyChatOverrides_EmptyParameters(t *testing.T) {
	pool := &Pool{}
	req := &schema.ChatRequest{Stream: true}
	extras := pool.ApplyChatOverrides(req)
	assert.Nil(t, extras)
	assert.True(t, req.Stream)
	assert.Nil(t, req.Temperature)
}

func TestMergeBodyExtras_TopLevelOverwrite(t *testing.T) {
	body := []byte(`{"model":"m","temperature":0.7,"top_k":1}`)
	extras := map[string]any{
		"top_k":    40,
		"mirostat": 2,
	}
	out, err := MergeBodyExtras(body, extras)
	assert.NoError(t, err)

	var m map[string]any
	assert.NoError(t, json.Unmarshal(out, &m))
	assert.Equal(t, "m", m["model"])
	assert.Equal(t, 0.7, m["temperature"])
	assert.EqualValues(t, 40, m["top_k"])
	assert.EqualValues(t, 2, m["mirostat"])
}

func TestMergeBodyExtras_EmptyExtras(t *testing.T) {
	body := []byte(`{"a":1}`)
	out, err := MergeBodyExtras(body, nil)
	assert.NoError(t, err)
	assert.Equal(t, string(body), string(out))
}

func TestMergeBodyExtras_NonObjectBody(t *testing.T) {
	_, err := MergeBodyExtras([]byte(`[1,2,3]`), map[string]any{"x": 1})
	assert.Error(t, err)
}
