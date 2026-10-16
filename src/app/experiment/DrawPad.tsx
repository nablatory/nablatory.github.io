import { useEffect, useRef, useState } from 'react';
import type { CustomInput } from '../../models/types';
import type { TensorPayload } from '../../runtime/trace';
import { Button } from '../../ui/controls';

/**
 * Draw or upload an image. Produces a [1, C, H, W] tensor with intensities in
 * [0, 1] at the model's input resolution; the model applies its own encoding.
 */
export function DrawPad({ spec, onUse }: { spec: CustomInput; onUse: (t: TensorPayload) => void }) {
  const { height: H, width: W, channels: C } = spec;
  const [pixels, setPixels] = useState(() => new Float32Array(C * H * W));
  const [dirty, setDirty] = useState(false);
  const drawing = useRef(false);
  const ref = useRef<HTMLCanvasElement>(null);
  const cell = Math.floor(168 / Math.max(H, W));

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext('2d')!;
    const img = new ImageData(W, H);
    for (let i = 0; i < H * W; i++) {
      for (let ch = 0; ch < 3; ch++) {
        const v = C === 3 ? pixels[ch * H * W + i] : pixels[i];
        img.data[i * 4 + ch] = Math.round(255 * (1 - v));
      }
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [pixels, H, W, C]);

  const paint = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * W;
    const y = ((e.clientY - r.top) / r.height) * H;
    setPixels((prev) => {
      const next = prev.slice();
      // Soft round brush of radius ≈ 1.4 px (MNIST strokes are ~2–3 px wide).
      for (let yy = Math.floor(y - 2); yy <= Math.ceil(y + 2); yy++)
        for (let xx = Math.floor(x - 2); xx <= Math.ceil(x + 2); xx++) {
          if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
          const d = Math.hypot(xx + 0.5 - x, yy + 0.5 - y);
          const ink = Math.max(0, Math.min(1, 1.6 - d));
          for (let ch = 0; ch < C; ch++) {
            const k = ch * H * W + yy * W + xx;
            next[k] = Math.max(next[k], ink);
          }
        }
      return next;
    });
    setDirty(true);
  };

  const upload = async (file: File) => {
    const bmp = await createImageBitmap(file);
    const off = new OffscreenCanvas(W, H);
    const ctx = off.getContext('2d')!;
    // Fit the whole image, preserving aspect ratio, on a white field.
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, W, H);
    const s = Math.min(W / bmp.width, H / bmp.height);
    ctx.drawImage(
      bmp,
      (W - bmp.width * s) / 2,
      (H - bmp.height * s) / 2,
      bmp.width * s,
      bmp.height * s,
    );
    const d = ctx.getImageData(0, 0, W, H).data;
    const next = new Float32Array(C * H * W);
    for (let i = 0; i < H * W; i++) {
      if (C === 3) {
        for (let ch = 0; ch < 3; ch++)
          next[ch * H * W + i] = spec.invertUploads ? 1 - d[i * 4 + ch] / 255 : d[i * 4 + ch] / 255;
      } else {
        const lum = (0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2]) / 255;
        next[i] = spec.invertUploads ? 1 - lum : lum;
      }
    }
    setPixels(next);
    setDirty(true);
  };

  return (
    <div className="drawpad">
      <canvas
        ref={ref}
        width={W}
        height={H}
        style={{ width: W * cell, height: H * cell }}
        className="drawpad-canvas"
        aria-label={`Drawing area, ${W} by ${H} pixels`}
        onPointerDown={(e) => {
          drawing.current = true;
          e.currentTarget.setPointerCapture(e.pointerId);
          paint(e);
        }}
        onPointerMove={(e) => drawing.current && paint(e)}
        onPointerUp={() => (drawing.current = false)}
      />
      <div className="drawpad-actions">
        <Button
          size="sm"
          variant="primary"
          disabled={!dirty}
          onClick={() => onUse({ shape: [1, C, H, W], data: pixels.slice() })}
        >
          Run on this image
        </Button>
        <Button
          size="sm"
          variant="quiet"
          onClick={() => {
            setPixels(new Float32Array(C * H * W));
            setDirty(false);
          }}
        >
          Clear
        </Button>
        <label className="btn btn-quiet btn-sm file-btn">
          Upload…
          <input
            type="file"
            accept="image/*"
            onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
          />
        </label>
      </div>
      <p className="field-hint">
        {W}×{H}, {C === 1 ? 'greyscale' : 'RGB'}. {spec.preprocessing} Uploads are resized in this
        browser{spec.invertUploads ? ' and inverted (dark ink → 1)' : ''}; nothing is sent anywhere.
      </p>
    </div>
  );
}
