// Command cli evaluates a Decision-Maker request JSON file against a running
// service and prints the response as JSON. Used by scripts/parity_golden.py
// as the "Go" producer in cross-language parity runs.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"decisionmaker"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "usage: cli <baseURL> <request.json>")
		os.Exit(2)
	}
	raw, err := os.ReadFile(os.Args[2])
	if err != nil {
		fmt.Fprintln(os.Stderr, "read request:", err)
		os.Exit(1)
	}
	var req decisionmaker.Request
	if err := json.Unmarshal(raw, &req); err != nil {
		fmt.Fprintln(os.Stderr, "decode request:", err)
		os.Exit(1)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	resp, err := decisionmaker.NewClient(os.Args[1], 0).Evaluate(ctx, &req)
	if err != nil {
		fmt.Fprintln(os.Stderr, "evaluate:", err)
		os.Exit(1)
	}
	out, err := json.MarshalIndent(resp, "", "  ")
	if err != nil {
		fmt.Fprintln(os.Stderr, "encode response:", err)
		os.Exit(1)
	}
	fmt.Println(string(out))
}
