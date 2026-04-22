package schema

import (
	"bufio"
	"bytes"
	"io"
	"strings"
)

// SSEEvent is a single parsed Server-Sent Events message.
type SSEEvent struct {
	Event string
	Data  []byte
	ID    string
}

// SSEScanner reads SSE-framed events from an io.Reader. Events are separated by blank lines; fields are prefixed with
// "event:", "data:", or "id:". Multiple data lines within an event are concatenated with newline.
type SSEScanner struct {
	r   *bufio.Reader
	err error
}

// NewSSEScanner wraps r with a bufio.Reader sized for typical SSE payloads.
func NewSSEScanner(r io.Reader) *SSEScanner {
	return &SSEScanner{r: bufio.NewReaderSize(r, 64*1024)}
}

// Next reads and returns the next SSE event. Returns io.EOF when the stream ends cleanly.
func (s *SSEScanner) Next() (*SSEEvent, error) {
	if s.err != nil {
		return nil, s.err
	}
	var (
		ev      SSEEvent
		data    bytes.Buffer
		haveAny bool
	)
	for {
		line, err := s.r.ReadString('\n')
		if err != nil && line == "" {
			s.err = err
			if haveAny {
				if data.Len() > 0 {
					ev.Data = bytes.TrimRight(data.Bytes(), "\n")
				}
				return &ev, nil
			}
			return nil, err
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			if !haveAny {
				continue
			}
			if data.Len() > 0 {
				ev.Data = bytes.TrimRight(data.Bytes(), "\n")
			}
			return &ev, nil
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		haveAny = true
		field, value, _ := strings.Cut(line, ":")
		value = strings.TrimPrefix(value, " ")
		switch field {
		case "event":
			ev.Event = value
		case "data":
			data.WriteString(value)
			data.WriteByte('\n')
		case "id":
			ev.ID = value
		}
	}
}
