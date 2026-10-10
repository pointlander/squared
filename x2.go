// Copyright 2026 The squared Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
)

const (
	x2Block  = 4
	x2Window = 8
	// x2Dim is one private block per position plus a shared band of the same width.
	x2Dim      = x2Block*x2Window + x2Block
	x2Hidden   = 64
	x2Vocab    = 256
	x2Inner    = 40
	x2InnerEta = 0.06
	x2Lambda   = 1e-4
	x2X0       = 0.05
	x2AdamEta  = 3e-3
	// x2FfnEta is the feedforward net's own gradient descent step.
	// Symbol embeddings use Adam at the caller's rate.
	x2FfnEta = 0.05
	x2Batch  = 16
	// x2Dropout is the probability of dropping an entry of X while training.
	// Kept entries are scaled by 1/(1-x2Dropout). Eval uses the full X.
	x2Dropout = 0.1
	// x2Holdout is the tail kept out of training and used for eval.
	x2Holdout = 8192
)

// x2Config sets the widths and the inner solve.
//
// A context of symbols is the matrix A. Column t carries that symbol's
// embedding twice: once in the private rows [t*block, (t+1)*block), and
// once in the shared band [window*block, window*block+block). The private
// blocks keep the order. The shared band makes the columns overlap, so the
// projector that solves X^2 A = A mixes one position with another. Gradient
// descent finds that X from a small multiple of the identity, shrinking X a
// little. That X is the feedforward network's embedding: the net reads it
// and scores the next symbol. While training, entries of X are dropped.
// The net is updated by its own gradient descent. Symbol embeddings are
// updated by Adam through the differentiated inner steps.
type x2Config struct {
	block, dim, window, hidden, vocab int
	inner                             int
	innerEta, lambda, x0              float64
}

func defaultX2Config() x2Config {
	return x2Config{
		block: x2Block, dim: x2Dim, window: x2Window, hidden: x2Hidden, vocab: x2Vocab,
		inner: x2Inner, innerEta: x2InnerEta, lambda: x2Lambda, x0: x2X0,
	}
}

// x2Model holds symbol embeddings and the feedforward network. An embedding
// has block coordinates. Embedding row vocab is the left-pad symbol and is
// never a target.
type x2Model struct {
	cfg x2Config
	emb []float64
	w1  []float64
	b1  []float64
	w2  []float64
	b2  []float64
}

func newX2Model(cfg x2Config, rng *rand.Rand) *x2Model {
	if cfg.block < 1 || cfg.window < 1 || cfg.dim != cfg.block*(cfg.window+1) {
		panic("x2 dim must equal block*(window+1)")
	}
	nIn := cfg.dim * cfg.dim
	m := &x2Model{
		cfg: cfg,
		emb: make([]float64, (cfg.vocab+1)*cfg.block),
		w1:  make([]float64, cfg.hidden*nIn),
		b1:  make([]float64, cfg.hidden),
		w2:  make([]float64, cfg.vocab*cfg.hidden),
		b2:  make([]float64, cfg.vocab),
	}
	for i := range m.emb {
		m.emb[i] = rng.NormFloat64() * 0.5
	}
	w1s := math.Sqrt(2 / float64(nIn))
	for i := range m.w1 {
		m.w1[i] = rng.NormFloat64() * w1s
	}
	for i := range m.w2 {
		m.w2[i] = rng.NormFloat64() * 0.01
	}
	return m
}

func (m *x2Model) pad() int { return m.cfg.vocab }

// x2Scratch is the working memory for one solve and its backward pass.
type x2Scratch struct {
	d, w, h, v, k, nIn int
	a, ahat            []float64
	norms              []float64
	x                  []float64
	xs                 []float64
	mm, bb, r          []float64
	t1                 []float64
	dt1, dr, dm, tmpDW []float64
	dat                []float64
	tmpDD              []float64
	dx, u              []float64
	daHat              []float64
	hpre, hid, dh      []float64
	logits, dlogits    []float64
	// xd is X after dropout, the embedding the feedforward net reads.
	// xMask is nil at eval. A dropped entry is 0 and a kept entry is 1/(1-p).
	xd, xMask []float64
}

func newX2Scratch(cfg x2Config) *x2Scratch {
	d, w, h, v := cfg.dim, cfg.window, cfg.hidden, cfg.vocab
	nIn := d * d
	dw, dd := d*w, d*d
	return &x2Scratch{
		d: d, w: w, h: h, v: v, k: cfg.inner, nIn: nIn,
		a: make([]float64, dw), ahat: make([]float64, dw), norms: make([]float64, w),
		x: make([]float64, dd), xs: make([]float64, (cfg.inner+1)*dd),
		mm: make([]float64, dw), bb: make([]float64, dw), r: make([]float64, dw),
		t1:    make([]float64, dw),
		dt1:   make([]float64, dw),
		dr:    make([]float64, dw),
		dm:    make([]float64, dw),
		tmpDW: make([]float64, dw),
		dat:   make([]float64, w*d),
		tmpDD: make([]float64, dd),
		dx:    make([]float64, dd),
		u:     make([]float64, dd),
		daHat: make([]float64, dw),
		hpre:  make([]float64, h), hid: make([]float64, h), dh: make([]float64, h),
		logits: make([]float64, v), dlogits: make([]float64, v),
	}
}

// mulInto sets dst = a @ b. a is m×n, b is n×p, dst is m×p, all row-major.
func mulInto(dst, a, b []float64, m, n, p int) {
	for i := range dst[:m*p] {
		dst[i] = 0
	}
	for i := 0; i < m; i++ {
		ar := a[i*n : i*n+n]
		dr := dst[i*p : i*p+p]
		for k := 0; k < n; k++ {
			aik := ar[k]
			if aik == 0 {
				continue
			}
			br := b[k*p : k*p+p]
			for j := 0; j < p; j++ {
				dr[j] += aik * br[j]
			}
		}
	}
}

// mulBTInto sets dst = a @ b^T. a is m×n, b is p×n, dst is m×p.
func mulBTInto(dst, a, b []float64, m, n, p int) {
	for i := 0; i < m; i++ {
		ar := a[i*n : i*n+n]
		dr := dst[i*p : i*p+p]
		for j := 0; j < p; j++ {
			br := b[j*n : j*n+n]
			s := 0.0
			for k := 0; k < n; k++ {
				s += ar[k] * br[k]
			}
			dr[j] = s
		}
	}
}

// mulATInto sets dst = a^T @ b. a is n×m, b is n×p, dst is m×p.
func mulATInto(dst, a, b []float64, n, m, p int) {
	for i := range dst[:m*p] {
		dst[i] = 0
	}
	for k := 0; k < n; k++ {
		ar := a[k*m : k*m+m]
		br := b[k*p : k*p+p]
		for i := 0; i < m; i++ {
			aik := ar[i]
			if aik == 0 {
				continue
			}
			dr := dst[i*p : i*p+p]
			for j := 0; j < p; j++ {
				dr[j] += aik * br[j]
			}
		}
	}
}

// normalizeCols writes unit-length columns of a into dst and their norms.
// A near-zero column is copied through with norm 0 so the backward pass can
// treat it as the identity map.
func normalizeCols(a, dst, norms []float64, d, w int) {
	for t := 0; t < w; t++ {
		n2 := 0.0
		for i := 0; i < d; i++ {
			v := a[i*w+t]
			n2 += v * v
		}
		n := math.Sqrt(n2)
		if n < 1e-8 {
			norms[t] = 0
			for i := 0; i < d; i++ {
				dst[i*w+t] = a[i*w+t]
			}
			continue
		}
		norms[t] = n
		inv := 1 / n
		for i := 0; i < d; i++ {
			dst[i*w+t] = a[i*w+t] * inv
		}
	}
}

// denormalizeCols maps a gradient on unit columns back to the raw columns.
func denormalizeCols(ahat, daHat, da, norms []float64, d, w int) {
	for t := 0; t < w; t++ {
		n := norms[t]
		if n < 1e-8 {
			for i := 0; i < d; i++ {
				da[i*w+t] = daHat[i*w+t]
			}
			continue
		}
		dot := 0.0
		for i := 0; i < d; i++ {
			dot += daHat[i*w+t] * ahat[i*w+t]
		}
		inv := 1 / n
		for i := 0; i < d; i++ {
			da[i*w+t] = daHat[i*w+t]*inv - dot*ahat[i*w+t]*inv
		}
	}
}

// forwardMats sets m = X A, b = X m, r = b - A and returns ||r||^2 and ||A||^2.
func (sc *x2Scratch) forwardMats(x, a []float64) (r2, a2 float64) {
	d, w := sc.d, sc.w
	mulInto(sc.mm, x, a, d, d, w)
	mulInto(sc.bb, x, sc.mm, d, d, w)
	for i := range a {
		r := sc.bb[i] - a[i]
		sc.r[i] = r
		r2 += r * r
		a2 += a[i] * a[i]
	}
	return r2, a2
}

// gdStep applies one gradient step of ||X^2 A - A||^2 + λ ||X||^2.
func (sc *x2Scratch) gdStep(x, a []float64, eta, lambda float64) {
	d, w := sc.d, sc.w
	sc.forwardMats(x, a)
	// G = 2 R M^T + 2 X^T R A^T + 2 λ X. sc.u holds the second term.
	mulBTInto(sc.tmpDD, sc.r, sc.mm, d, w, d)
	mulATInto(sc.t1, x, sc.r, d, d, w)
	mulBTInto(sc.u, sc.t1, a, d, w, d)
	for i := range x {
		g := 2*sc.tmpDD[i] + 2*sc.u[i] + 2*lambda*x[i]
		x[i] -= eta * g
	}
}

// vjp backpropagates u = ∂L/∂X_after through one gradient step.
// dx receives ∂L/∂X_before. da is accumulated with ∂L/∂A.
func (sc *x2Scratch) vjp(x, a, u, dx, da []float64, eta, lambda float64) {
	d, w := sc.d, sc.w
	sc.forwardMats(x, a)
	mulATInto(sc.t1, x, sc.r, d, d, w)

	// X2 = X - η G, G = 2 G1 + 2 G2 + 2 λ X, dG = -η U.
	scale := -2 * eta
	for i := range dx {
		dx[i] = (1 - 2*lambda*eta) * u[i]
	}

	// G2 = T1 @ A^T, T1 = X^T @ R, dG2 = scale U.
	mulInto(sc.dt1, u, a, d, d, w)
	for i := range sc.dt1 {
		sc.dt1[i] *= scale
	}
	mulATInto(sc.dat, sc.t1, u, d, w, d)
	for i := 0; i < w; i++ {
		for j := 0; j < d; j++ {
			da[j*w+i] += scale * sc.dat[i*d+j]
		}
	}
	mulBTInto(sc.tmpDD, sc.r, sc.dt1, d, w, d)
	for i := range dx {
		dx[i] += sc.tmpDD[i]
	}
	mulInto(sc.dr, x, sc.dt1, d, d, w)

	// G1 = R @ M^T, dG1 = scale U.
	mulInto(sc.tmpDW, u, sc.mm, d, d, w)
	for i := range sc.dr {
		sc.dr[i] += scale * sc.tmpDW[i]
	}
	mulATInto(sc.dm, u, sc.r, d, d, w)
	for i := range sc.dm {
		sc.dm[i] *= scale
	}

	// R = X M - A.
	for i := range da {
		da[i] -= sc.dr[i]
	}
	mulBTInto(sc.tmpDD, sc.dr, sc.mm, d, w, d)
	for i := range dx {
		dx[i] += sc.tmpDD[i]
	}
	mulATInto(sc.tmpDW, x, sc.dr, d, d, w)
	for i := range sc.dm {
		sc.dm[i] += sc.tmpDW[i]
	}

	// M = X A.
	mulBTInto(sc.tmpDD, sc.dm, a, d, w, d)
	for i := range dx {
		dx[i] += sc.tmpDD[i]
	}
	mulATInto(sc.tmpDW, x, sc.dm, d, d, w)
	for i := range da {
		da[i] += sc.tmpDW[i]
	}
}

// solve replaces x with K steps of gradient descent on X^2 A = A.
// a is the already normalized context matrix. The trajectory is kept in sc.xs.
// The relative residual ||X^2 A - A|| / ||A|| is returned.
func (sc *x2Scratch) solve(a []float64, eta, lambda, x0 float64) float64 {
	d, k := sc.d, sc.k
	nIn := d * d
	for i := range sc.x {
		sc.x[i] = 0
	}
	for i := 0; i < d; i++ {
		sc.x[i*d+i] = x0
	}
	copy(sc.xs[:nIn], sc.x)
	for step := 0; step < k; step++ {
		sc.gdStep(sc.x, a, eta, lambda)
		copy(sc.xs[(step+1)*nIn:(step+2)*nIn], sc.x)
	}
	r2, a2 := sc.forwardMats(sc.x, a)
	return math.Sqrt(r2 / (a2 + 1e-12))
}

// writeContextColumns builds A. Column t holds that symbol's embedding in
// private rows [t*block, (t+1)*block) and again in the shared band
// [window*block, window*block+block). Every other private block stays zero.
func writeContextColumns(emb []float64, block, window int, ctx []int, a []float64) {
	for i := range a {
		a[i] = 0
	}
	shared := window * block
	for t := 0; t < window; t++ {
		e := emb[ctx[t]*block : ctx[t]*block+block]
		row := t * block
		for i := 0; i < block; i++ {
			a[(row+i)*window+t] = e[i]
			a[(shared+i)*window+t] = e[i]
		}
	}
}

// addColumnGrads adds ∂L/∂A into the symbol embeddings. Each column holds the
// embedding in its private block and in the shared band, so both contribute.
func addColumnGrads(gEmb, da []float64, ctx []int, block, window int) {
	shared := window * block
	for t := 0; t < window; t++ {
		ge := gEmb[ctx[t]*block : ctx[t]*block+block]
		row := t * block
		for i := 0; i < block; i++ {
			ge[i] += da[(row+i)*window+t]
			ge[i] += da[(shared+i)*window+t]
		}
	}
}

func (m *x2Model) buildA(ctx []int, a []float64) {
	writeContextColumns(m.emb, m.cfg.block, m.cfg.window, ctx, a)
}

func forwardNet(w1, b1, w2, b2, x []float64, nIn, h, v int, hpre, hid, logits []float64) {
	for i := 0; i < h; i++ {
		s := b1[i]
		row := w1[i*nIn : i*nIn+nIn]
		for j := 0; j < nIn; j++ {
			s += row[j] * x[j]
		}
		hpre[i] = s
		if s > 0 {
			hid[i] = s
		} else {
			hid[i] = 0
		}
	}
	for i := 0; i < v; i++ {
		s := b2[i]
		row := w2[i*h : i*h+h]
		for j := 0; j < h; j++ {
			s += row[j] * hid[j]
		}
		logits[i] = s
	}
}

func backwardNet(w1, w2, x []float64, nIn, h, v int, hpre, hid, dh, dlogits, gW1, gB1, gW2, gB2, dx []float64) {
	for i := range dh {
		dh[i] = 0
	}
	for i := range dx {
		dx[i] = 0
	}
	for i := 0; i < v; i++ {
		g := dlogits[i]
		if g == 0 {
			continue
		}
		gB2[i] += g
		row := w2[i*h : i*h+h]
		grow := gW2[i*h : i*h+h]
		for j := 0; j < h; j++ {
			grow[j] += g * hid[j]
			dh[j] += row[j] * g
		}
	}
	for i := 0; i < h; i++ {
		if hpre[i] <= 0 {
			dh[i] = 0
		}
	}
	for i := 0; i < h; i++ {
		g := dh[i]
		if g == 0 {
			continue
		}
		gB1[i] += g
		row := w1[i*nIn : i*nIn+nIn]
		grow := gW1[i*nIn : i*nIn+nIn]
		for j := 0; j < nIn; j++ {
			grow[j] += g * x[j]
			dx[j] += row[j] * g
		}
	}
}

func (m *x2Model) ffn(x []float64, sc *x2Scratch) {
	forwardNet(m.w1, m.b1, m.w2, m.b2, x, sc.nIn, sc.h, sc.v, sc.hpre, sc.hid, sc.logits)
}

// crossEntropy writes softmax(logits) - one_hot(y) into dlogits and returns
// the cross-entropy.
func crossEntropy(logits, dlogits []float64, y int) float64 {
	max := logits[0]
	for _, v := range logits[1:] {
		if v > max {
			max = v
		}
	}
	sum := 0.0
	for i, v := range logits {
		e := math.Exp(v - max)
		dlogits[i] = e
		sum += e
	}
	inv := 1 / sum
	for i := range dlogits {
		dlogits[i] *= inv
	}
	loss := max + math.Log(sum) - logits[y]
	dlogits[y] -= 1
	return loss
}

// ffnBackward accumulates parameter gradients and writes ∂L/∂X into u.
func (m *x2Model) ffnBackward(x []float64, gW1, gB1, gW2, gB2, u []float64, sc *x2Scratch) {
	backwardNet(m.w1, m.w2, x, sc.nIn, sc.h, sc.v, sc.hpre, sc.hid, sc.dh, sc.dlogits, gW1, gB1, gW2, gB2, u)
}

// ffnInput is the solved X. While training, xMask drops entries. That matrix
// is the feedforward net's embedding.
func (sc *x2Scratch) ffnInput() []float64 {
	if sc.xMask == nil {
		return sc.x
	}
	if len(sc.xd) != len(sc.x) {
		sc.xd = make([]float64, len(sc.x))
	}
	for i := range sc.x {
		sc.xd[i] = sc.xMask[i] * sc.x[i]
	}
	return sc.xd
}

// exampleGrad runs one context and adds its gradients. Grad buffers are not
// cleared. loss is the cross-entropy and rel is the constraint residual.
func (m *x2Model) exampleGrad(ctx []int, y int, gEmb, gW1, gB1, gW2, gB2 []float64, sc *x2Scratch) (loss, rel float64) {
	cfg := m.cfg
	d, w, block := cfg.dim, cfg.window, cfg.block
	m.buildA(ctx, sc.a)
	normalizeCols(sc.a, sc.ahat, sc.norms, d, w)
	rel = sc.solve(sc.ahat, cfg.innerEta, cfg.lambda, cfg.x0)
	xIn := sc.ffnInput()
	m.ffn(xIn, sc)
	loss = crossEntropy(sc.logits, sc.dlogits, y)
	m.ffnBackward(xIn, gW1, gB1, gW2, gB2, sc.u, sc)
	if sc.xMask != nil {
		for i := range sc.u {
			sc.u[i] *= sc.xMask[i]
		}
	}

	for i := range sc.daHat {
		sc.daHat[i] = 0
	}
	nIn := sc.nIn
	for step := sc.k - 1; step >= 0; step-- {
		xBefore := sc.xs[step*nIn : (step+1)*nIn]
		for i := range sc.dx {
			sc.dx[i] = 0
		}
		sc.vjp(xBefore, sc.ahat, sc.u, sc.dx, sc.daHat, cfg.innerEta, cfg.lambda)
		copy(sc.u, sc.dx)
	}
	denormalizeCols(sc.ahat, sc.daHat, sc.a, sc.norms, d, w)
	addColumnGrads(gEmb, sc.a, ctx, block, w)
	return loss, rel
}

// logits scores the next symbol for a context. The result is scratch memory
// and is invalid after the next call that uses sc.
func (m *x2Model) logits(ctx []int, sc *x2Scratch) []float64 {
	m.buildA(ctx, sc.a)
	normalizeCols(sc.a, sc.ahat, sc.norms, m.cfg.dim, m.cfg.window)
	sc.solve(sc.ahat, m.cfg.innerEta, m.cfg.lambda, m.cfg.x0)
	m.ffn(sc.ffnInput(), sc)
	return sc.logits
}

func x2Context(text []byte, pos, window, pad int) []int {
	ctx := make([]int, window)
	for t := 0; t < window; t++ {
		i := pos - window + t
		if i < 0 || i >= len(text) {
			ctx[t] = pad
		} else {
			ctx[t] = int(text[i])
		}
	}
	return ctx
}

// sampleXDropout draws a fresh mask for one training example. A nil mask
// leaves X unchanged, which is how eval and generation run.
func sampleXDropout(sc *x2Scratch, rng *rand.Rand) {
	if x2Dropout <= 0 || rng == nil {
		sc.xMask = nil
		return
	}
	if len(sc.xMask) != sc.nIn {
		sc.xMask = make([]float64, sc.nIn)
	}
	keep := 1 / (1 - x2Dropout)
	for i := range sc.xMask {
		if rng.Float64() < x2Dropout {
			sc.xMask[i] = 0
		} else {
			sc.xMask[i] = keep
		}
	}
}

// gdUpdate applies one gradient descent step.
func gdUpdate(p, g []float64, eta float64) {
	for i := range p {
		p[i] -= eta * g[i]
	}
}

func samplePositions(n, count int, rng *rand.Rand) []int {
	if n <= 0 || count <= 0 {
		return nil
	}
	pos := make([]int, count)
	for i := range pos {
		pos[i] = rng.Intn(n)
	}
	return pos
}

func clipGrad(gs [][]float64, max float64) {
	n2 := 0.0
	for _, g := range gs {
		for _, v := range g {
			n2 += v * v
		}
	}
	n := math.Sqrt(n2)
	if n <= max || n == 0 {
		return
	}
	s := max / n
	for _, g := range gs {
		for i := range g {
			g[i] *= s
		}
	}
}

func adamStep(p, g, m, v []float64, t int, eta float64) {
	const b1, b2 = 0.9, 0.999
	c1 := 1 - math.Pow(b1, float64(t))
	c2 := 1 - math.Pow(b2, float64(t))
	for i := range p {
		m[i] = b1*m[i] + (1-b1)*g[i]
		v[i] = b2*v[i] + (1-b2)*g[i]*g[i]
		p[i] -= eta * (m[i] / c1) / (math.Sqrt(v[i]/c2) + 1e-8)
	}
}

// train updates the symbol embeddings with Adam and the feedforward net with
// its own gradient step, on random windows of text. logEvery prints progress
// to stderr; 0 stays quiet. The returned loss and residual are means over the
// last batch.
func (m *x2Model) train(text []byte, steps, batch int, eta float64, rng *rand.Rand, logEvery int) (loss, residual float64) {
	if len(text) == 0 || steps == 0 {
		return 0, 0
	}
	pos := samplePositions(len(text), steps*batch, rng)
	return m.trainAt(text, pos, batch, eta, rng, logEvery)
}

// trainAt runs one step of Adam on the symbol embeddings and a separate
// gradient step on the feedforward net. Positions are shared with the column
// baseline so the two losses are comparable.
func (m *x2Model) trainAt(text []byte, pos []int, batch int, eta float64, rng *rand.Rand, logEvery int) (loss, residual float64) {
	if len(text) == 0 || len(pos) == 0 || batch < 1 {
		return 0, 0
	}
	steps := len(pos) / batch
	sc := newX2Scratch(m.cfg)
	gEmb := make([]float64, len(m.emb))
	gW1 := make([]float64, len(m.w1))
	gB1 := make([]float64, len(m.b1))
	gW2 := make([]float64, len(m.w2))
	gB2 := make([]float64, len(m.b2))
	mEmb := make([]float64, len(m.emb))
	vEmb := make([]float64, len(m.emb))
	grads := [][]float64{gEmb, gW1, gB1, gW2, gB2}
	ffnGrads := [][]float64{gW1, gB1, gW2, gB2}
	pad := m.pad()
	window := m.cfg.window
	for step := 1; step <= steps; step++ {
		for _, g := range grads {
			for i := range g {
				g[i] = 0
			}
		}
		loss, residual = 0, 0
		correct := 0
		base := (step - 1) * batch
		for i := 0; i < batch; i++ {
			p := pos[base+i]
			ctx := x2Context(text, p, window, pad)
			y := int(text[p])
			sampleXDropout(sc, rng)
			l, rel := m.exampleGrad(ctx, y, gEmb, gW1, gB1, gW2, gB2, sc)
			loss += l
			residual += rel
			if !math.IsNaN(l) && argmax(sc.logits) == y {
				correct++
			}
		}
		inv := 1 / float64(batch)
		loss *= inv
		residual *= inv
		for _, g := range grads {
			for i := range g {
				g[i] *= inv
			}
		}
		if math.IsNaN(loss) || math.IsInf(loss, 0) {
			return loss, residual
		}
		clipGrad([][]float64{gEmb}, 1)
		clipGrad(ffnGrads, 1)
		adamStep(m.emb, gEmb, mEmb, vEmb, step, eta)
		gdUpdate(m.w1, gW1, x2FfnEta)
		gdUpdate(m.b1, gB1, x2FfnEta)
		gdUpdate(m.w2, gW2, x2FfnEta)
		gdUpdate(m.b2, gB2, x2FfnEta)
		if logEvery > 0 && (step%logEvery == 0 || step == 1 || step == steps) {
			fmt.Fprintf(os.Stderr, "x2 step=%d loss=%.4f residual=%.4f acc=%.3f\n",
				step, loss, residual, float64(correct)/float64(batch))
		}
	}
	return loss, residual
}

func argmax(v []float64) int {
	best, arg := v[0], 0
	for i := 1; i < len(v); i++ {
		if v[i] > best {
			best = v[i]
			arg = i
		}
	}
	return arg
}

func sampleLogits(rng *rand.Rand, logits []float64) int {
	max := logits[0]
	for _, v := range logits[1:] {
		if v > max {
			max = v
		}
	}
	sum := 0.0
	probs := make([]float64, len(logits))
	for i, v := range logits {
		e := math.Exp(v - max)
		probs[i] = e
		sum += e
	}
	x := rng.Float64() * sum
	acc := 0.0
	for i, p := range probs {
		acc += p
		if x < acc {
			return i
		}
	}
	return len(probs) - 1
}

// x2Alphabet is the set of bytes that occur in the corpus, in order.
type x2Alphabet struct {
	to    []int
	from  []byte
	space int
}

func newX2Alphabet(text []byte) x2Alphabet {
	seen := make([]bool, 256)
	for _, b := range text {
		seen[b] = true
	}
	a := x2Alphabet{to: make([]int, 256)}
	for i := range a.to {
		a.to[i] = -1
	}
	for b := 0; b < 256; b++ {
		if seen[b] {
			a.to[b] = len(a.from)
			a.from = append(a.from, byte(b))
		}
	}
	if id := a.to[' ']; id >= 0 {
		a.space = id
	}
	return a
}

// encode maps bytes to symbol ids. A byte that never occurs in the corpus
// uses the space id, or 0 when the corpus has no space.
func (a x2Alphabet) encode(text []byte) []byte {
	out := make([]byte, len(text))
	for i, b := range text {
		id := a.to[b]
		if id < 0 {
			id = a.space
		}
		out[i] = byte(id)
	}
	return out
}

func (a x2Alphabet) decode(ids []byte) []byte {
	out := make([]byte, len(ids))
	for i, id := range ids {
		out[i] = a.from[id]
	}
	return out
}

// meanLossAt is the mean cross-entropy at the given positions.
func (m *x2Model) meanLossAt(text []byte, pos []int) float64 {
	if len(text) == 0 || len(pos) == 0 {
		return 0
	}
	sc := newX2Scratch(m.cfg)
	sum := 0.0
	pad := m.pad()
	window := m.cfg.window
	for _, p := range pos {
		ctx := x2Context(text, p, window, pad)
		sum += crossEntropy(m.logits(ctx, sc), sc.dlogits, int(text[p]))
	}
	return sum / float64(len(pos))
}

// generate continues prompt by n symbols drawn from the model.
func (m *x2Model) generate(prompt []byte, n int, rng *rand.Rand) []byte {
	out := append([]byte(nil), prompt...)
	sc := newX2Scratch(m.cfg)
	pad := m.pad()
	window := m.cfg.window
	for range n {
		ctx := x2Context(out, len(out), window, pad)
		logits := m.logits(ctx, sc)
		next := sampleLogits(rng, logits)
		out = append(out, byte(next))
	}
	return out
}

// x2Split keeps a tail of text out of training. A short buffer is all training.
func x2Split(text []byte, window int) (train, eval []byte) {
	hold := x2Holdout
	if hold > len(text)/5 {
		hold = len(text) / 5
	}
	if hold < window || hold >= len(text) {
		return text, nil
	}
	return text[:len(text)-hold], text[len(text)-hold:]
}

// stripGutenberg drops the Project Gutenberg header and footer when the
// standard markers are present. Text without them is returned unchanged.
func stripGutenberg(s string) string {
	const start = "*** START OF THE PROJECT GUTENBERG EBOOK"
	const end = "*** END OF THE PROJECT GUTENBERG EBOOK"
	if i := strings.Index(s, start); i >= 0 {
		rest := s[i+len(start):]
		if nl := strings.IndexByte(rest, '\n'); nl >= 0 {
			s = rest[nl+1:]
		} else {
			s = rest
		}
	}
	if i := strings.Index(s, end); i >= 0 {
		s = s[:i]
	}
	return s
}

// colModel is the feedforward net on the normalized columns of A. It skips
// the X^2 A = A solve. Its held-out loss is the baseline for that constraint.
// Hidden width, embedding size, and the Adam step match the constraint model.
// The first layer reads dim*window inputs, the concatenated columns.
type colModel struct {
	cfg x2Config
	emb []float64
	w1  []float64
	b1  []float64
	w2  []float64
	b2  []float64
}

func newColModel(cfg x2Config, rng *rand.Rand) *colModel {
	if cfg.block < 1 || cfg.window < 1 || cfg.dim != cfg.block*(cfg.window+1) {
		panic("x2 dim must equal block*(window+1)")
	}
	nIn := cfg.dim * cfg.window
	m := &colModel{
		cfg: cfg,
		emb: make([]float64, (cfg.vocab+1)*cfg.block),
		w1:  make([]float64, cfg.hidden*nIn),
		b1:  make([]float64, cfg.hidden),
		w2:  make([]float64, cfg.vocab*cfg.hidden),
		b2:  make([]float64, cfg.vocab),
	}
	for i := range m.emb {
		m.emb[i] = rng.NormFloat64() * 0.5
	}
	w1s := math.Sqrt(2 / float64(nIn))
	for i := range m.w1 {
		m.w1[i] = rng.NormFloat64() * w1s
	}
	for i := range m.w2 {
		m.w2[i] = rng.NormFloat64() * 0.01
	}
	return m
}

func (m *colModel) logits(ctx []int, sc *x2Scratch) []float64 {
	d, w := m.cfg.dim, m.cfg.window
	writeContextColumns(m.emb, m.cfg.block, w, ctx, sc.a)
	normalizeCols(sc.a, sc.ahat, sc.norms, d, w)
	forwardNet(m.w1, m.b1, m.w2, m.b2, sc.ahat, d*w, sc.h, sc.v, sc.hpre, sc.hid, sc.logits)
	return sc.logits
}

// exampleGrad adds one context's gradients. Grad buffers are not cleared.
func (m *colModel) exampleGrad(ctx []int, y int, gEmb, gW1, gB1, gW2, gB2 []float64, sc *x2Scratch) float64 {
	d, w, block := m.cfg.dim, m.cfg.window, m.cfg.block
	writeContextColumns(m.emb, block, w, ctx, sc.a)
	normalizeCols(sc.a, sc.ahat, sc.norms, d, w)
	nIn := d * w
	forwardNet(m.w1, m.b1, m.w2, m.b2, sc.ahat, nIn, sc.h, sc.v, sc.hpre, sc.hid, sc.logits)
	loss := crossEntropy(sc.logits, sc.dlogits, y)
	backwardNet(m.w1, m.w2, sc.ahat, nIn, sc.h, sc.v, sc.hpre, sc.hid, sc.dh, sc.dlogits, gW1, gB1, gW2, gB2, sc.u)
	denormalizeCols(sc.ahat, sc.u, sc.a, sc.norms, d, w)
	addColumnGrads(gEmb, sc.a, ctx, block, w)
	return loss
}

func (m *colModel) trainAt(text []byte, pos []int, batch int, eta float64, logEvery int) float64 {
	if len(text) == 0 || len(pos) == 0 || batch < 1 {
		return 0
	}
	steps := len(pos) / batch
	sc := newX2Scratch(m.cfg)
	gEmb := make([]float64, len(m.emb))
	gW1 := make([]float64, len(m.w1))
	gB1 := make([]float64, len(m.b1))
	gW2 := make([]float64, len(m.w2))
	gB2 := make([]float64, len(m.b2))
	mEmb := make([]float64, len(m.emb))
	vEmb := make([]float64, len(m.emb))
	grads := [][]float64{gEmb, gW1, gB1, gW2, gB2}
	ffnGrads := [][]float64{gW1, gB1, gW2, gB2}
	pad := m.cfg.vocab
	window := m.cfg.window
	var loss float64
	for step := 1; step <= steps; step++ {
		for _, g := range grads {
			for i := range g {
				g[i] = 0
			}
		}
		loss = 0
		base := (step - 1) * batch
		for i := 0; i < batch; i++ {
			p := pos[base+i]
			ctx := x2Context(text, p, window, pad)
			loss += m.exampleGrad(ctx, int(text[p]), gEmb, gW1, gB1, gW2, gB2, sc)
		}
		inv := 1 / float64(batch)
		loss *= inv
		for _, g := range grads {
			for i := range g {
				g[i] *= inv
			}
		}
		if math.IsNaN(loss) || math.IsInf(loss, 0) {
			return loss
		}
		clipGrad([][]float64{gEmb}, 1)
		clipGrad(ffnGrads, 1)
		adamStep(m.emb, gEmb, mEmb, vEmb, step, eta)
		gdUpdate(m.w1, gW1, x2FfnEta)
		gdUpdate(m.b1, gB1, x2FfnEta)
		gdUpdate(m.w2, gW2, x2FfnEta)
		gdUpdate(m.b2, gB2, x2FfnEta)
		if logEvery > 0 && (step%logEvery == 0 || step == 1 || step == steps) {
			fmt.Fprintf(os.Stderr, "cols step=%d loss=%.4f\n", step, loss)
		}
	}
	return loss
}

func (m *colModel) meanLossAt(text []byte, pos []int) float64 {
	if len(text) == 0 || len(pos) == 0 {
		return 0
	}
	sc := newX2Scratch(m.cfg)
	sum := 0.0
	pad := m.cfg.vocab
	window := m.cfg.window
	for _, p := range pos {
		ctx := x2Context(text, p, window, pad)
		sum += crossEntropy(m.logits(ctx, sc), sc.dlogits, int(text[p]))
	}
	return sum / float64(len(pos))
}

func runX2(file, promptText string, n int, seed int64, steps int) error {
	if n < 0 {
		return fmt.Errorf("-n must be >= 0")
	}
	if steps < 0 {
		return fmt.Errorf("-steps must be >= 0")
	}
	if promptText == "" {
		return fmt.Errorf("prompt has no symbols")
	}
	data, err := os.ReadFile(file)
	if err != nil {
		return err
	}
	text := []byte(stripGutenberg(string(data)))
	if len(text) == 0 {
		return fmt.Errorf("%s is empty", file)
	}
	cfg := defaultX2Config()
	trainText, evalText := x2Split(text, cfg.window)
	alpha := newX2Alphabet(trainText)
	cfg.vocab = len(alpha.from)
	rng := rand.New(rand.NewSource(seed))
	m := newX2Model(cfg, rng)
	trainSym := alpha.encode(trainText)
	pos := samplePositions(len(trainSym), steps*x2Batch, rng)
	_, residual := m.trainAt(trainSym, pos, x2Batch, x2AdamEta, rng, 50)
	evalSym := alpha.encode(evalText)
	if len(evalSym) == 0 {
		evalSym = trainSym
	}
	evalPos := samplePositions(len(evalSym), 64, rng)
	loss := m.meanLossAt(evalSym, evalPos)
	// The column net is seeded apart from the constraint model so it does not
	// consume the positions or the eval sample drawn above.
	cols := newColModel(cfg, rand.New(rand.NewSource(seed+1)))
	cols.trainAt(trainSym, pos, x2Batch, x2AdamEta, 0)
	colLoss := cols.meanLossAt(evalSym, evalPos)
	out := alpha.decode(m.generate(alpha.encode([]byte(promptText)), n, rng))
	fmt.Fprintf(os.Stderr, "x2 bytes=%d heldout=%d vocab=%d block=%d window=%d dim=%d drop=%.2f steps=%d eval=%.4f cols=%.4f residual=%.4f seed=%d\n",
		len(text), len(evalText), cfg.vocab, m.cfg.block, m.cfg.window, m.cfg.dim, x2Dropout, steps, loss, colLoss, residual, seed)
	fmt.Println(string(out))
	return nil
}
