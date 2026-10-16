import { useMemo } from 'react';
import { lutFor, palette, type Scale } from './colormap';
import { fmt } from '../ui/format';

/** Colour legend that states the exact range mapped. */
export function ScaleBar({
  scale,
  label,
  width = 140,
}: {
  scale: Scale;
  label?: string;
  width?: number;
}) {
  const gradient = useMemo(() => {
    const lut = lutFor(palette(), scale.mode);
    const stops: string[] = [];
    for (let i = 0; i <= 8; i++) {
      const k = Math.round((i / 8) * 255);
      stops.push(`rgb(${lut[k * 3]},${lut[k * 3 + 1]},${lut[k * 3 + 2]}) ${(i / 8) * 100}%`);
    }
    return `linear-gradient(90deg, ${stops.join(',')})`;
  }, [scale.mode]);
  return (
    <span className="scalebar" style={{ width }}>
      <span className="scalebar-strip" style={{ background: gradient }} />
      <span className="scalebar-ticks">
        <span>{fmt(scale.min, 3)}</span>
        {scale.mode === 'diverging' && <span>0</span>}
        <span>{fmt(scale.max, 3)}</span>
      </span>
      {label && <span className="scalebar-label">{label}</span>}
    </span>
  );
}
