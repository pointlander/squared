// Copyright 2026 The squared Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command squared trains a byte model and continues a prompt.
//
// Each distinct byte is a symbol. Column t of A is that symbol's embedding
// written into its own coordinates and scaled to unit length, so the solved
// X keeps the order of the window. Gradient descent solves X^2 A = A, and a
// feedforward network maps X to the next symbol. -n counts bytes.
//
//	squared -prompt "To be, or not to be" -n 80
package main

import (
	"flag"
	"fmt"
	"os"
)

func main() {
	file := flag.String("file", "pg100.txt", "corpus file")
	promptFlag := flag.String("prompt", "To be, or not to be", "prompt to continue")
	n := flag.Int("n", 80, "bytes to generate after the prompt")
	seed := flag.Int64("seed", 1, "random seed")
	steps := flag.Int("steps", 400, "Adam steps")
	flag.Parse()

	if err := runX2(*file, *promptFlag, *n, *seed, *steps); err != nil {
		fmt.Fprintf(os.Stderr, "squared: %s\n", err)
		os.Exit(1)
	}
}
