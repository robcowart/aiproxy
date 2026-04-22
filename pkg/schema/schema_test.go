package schema

import (
	"encoding/json"
	"io"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

type nopCloser struct{ io.Reader }

func (nopCloser) Close() error { return nil }

func TestNormalizeStreamOptions(t *testing.T) {
	cases := []struct {
		name          string
		in            json.RawMessage
		wantOptedOut  bool
		wantInclude   bool
		wantPreserves string
		wantErr       bool
	}{
		{name: "nil", in: nil, wantInclude: true},
		{name: "empty", in: json.RawMessage(``), wantInclude: true},
		{name: "null", in: json.RawMessage(`null`), wantInclude: true},
		{name: "empty_object", in: json.RawMessage(`{}`), wantInclude: true},
		{name: "absent_field", in: json.RawMessage(`{"other":"x"}`), wantInclude: true, wantPreserves: `"other":"x"`},
		{name: "explicit_true", in: json.RawMessage(`{"include_usage":true}`), wantInclude: true},
		{name: "explicit_false", in: json.RawMessage(`{"include_usage":false}`), wantInclude: true, wantOptedOut: true},
		{name: "false_with_other", in: json.RawMessage(`{"include_usage":false,"other":42}`), wantInclude: true, wantOptedOut: true, wantPreserves: `"other":42`},
		{name: "not_object", in: json.RawMessage(`"nope"`), wantErr: true},
		{name: "bad_type", in: json.RawMessage(`{"include_usage":"yes"}`), wantErr: true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			out, optedOut, err := NormalizeStreamOptions(c.in)
			if c.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.Equal(t, c.wantOptedOut, optedOut)
			if c.wantInclude {
				assert.Contains(t, string(out), `"include_usage":true`)
			}
			if c.wantPreserves != "" {
				assert.Contains(t, string(out), c.wantPreserves)
			}
		})
	}
}

func TestRegistry_Defaults(t *testing.T) {
	r := NewRegistry()
	for _, name := range []string{"openai", "llamacpp", "anthropic", "google"} {
		tr, err := r.Get(name)
		assert.NoError(t, err, name)
		assert.Equal(t, name, tr.Name())
	}
	_, err := r.Get("bogus")
	assert.Error(t, err)
}

func TestSSEScanner(t *testing.T) {
	input := "event: foo\ndata: bar\ndata: baz\n\nevent: done\ndata: x\n\n"
	s := NewSSEScanner(strings.NewReader(input))

	ev, err := s.Next()
	assert.NoError(t, err)
	assert.Equal(t, "foo", ev.Event)
	assert.Equal(t, "bar\nbaz", string(ev.Data))

	ev, err = s.Next()
	assert.NoError(t, err)
	assert.Equal(t, "done", ev.Event)
	assert.Equal(t, "x", string(ev.Data))

	_, err = s.Next()
	assert.Equal(t, io.EOF, err)
}

func TestDecodeStringOrArray(t *testing.T) {
	v, err := decodeStringOrArray(json.RawMessage(`"hello"`))
	assert.NoError(t, err)
	assert.Equal(t, []string{"hello"}, v)

	v, err = decodeStringOrArray(json.RawMessage(`["a","b"]`))
	assert.NoError(t, err)
	assert.Equal(t, []string{"a", "b"}, v)

	v, err = decodeStringOrArray(nil)
	assert.NoError(t, err)
	assert.Nil(t, v)

	_, err = decodeStringOrArray(json.RawMessage(`123`))
	assert.Error(t, err)
}

func TestMessageContentToText(t *testing.T) {
	s, err := messageContentToText(json.RawMessage(`"hello"`))
	assert.NoError(t, err)
	assert.Equal(t, "hello", s)

	s, err = messageContentToText(json.RawMessage(`[{"type":"text","text":"a"},{"type":"text","text":"b"}]`))
	assert.NoError(t, err)
	assert.Equal(t, "ab", s)

	s, err = messageContentToText(nil)
	assert.NoError(t, err)
	assert.Equal(t, "", s)
}
