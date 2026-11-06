import { useCallback, useRef, useState } from 'react'
import type { KeyboardEvent, PointerEvent } from 'react'

const clamp = (value: number) => Math.min(100, Math.max(0, value))

export type ComparisonSliderProps = {
  beforeSrc: string
  afterSrc: string
  beforeLabel?: string
  afterLabel?: string
  /** Initial divider position as a percentage of the width. */
  initialPosition?: number
  /** Width / height of the compared images. Defaults to 4:3. */
  aspectRatio?: number
  /** `cover` fills the frame (placeholders); `contain` never crops a photo. */
  fit?: 'cover' | 'contain'
  className?: string
}

/**
 * Reusable before/after comparison slider.
 *
 * Drag anywhere on the image, or focus the handle and use the arrow keys.
 * Vertical swipes still scroll the page on touch devices.
 */
export function ComparisonSlider({
  beforeSrc,
  afterSrc,
  beforeLabel = 'Original',
  afterLabel = 'Enhanced',
  initialPosition = 50,
  aspectRatio,
  fit = 'cover',
  className = '',
}: ComparisonSliderProps) {
  const [position, setPosition] = useState(() => clamp(initialPosition))
  const regionRef = useRef<HTMLDivElement>(null)
  const draggingRef = useRef(false)

  const updateFromClientX = useCallback((clientX: number) => {
    const region = regionRef.current
    if (!region) return

    const { left, width } = region.getBoundingClientRect()
    if (width === 0) return

    setPosition(clamp(((clientX - left) / width) * 100))
  }, [])

  const handlePointerDown = (event: PointerEvent<HTMLDivElement>) => {
    draggingRef.current = true
    event.currentTarget.setPointerCapture(event.pointerId)
    updateFromClientX(event.clientX)
  }

  const handlePointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (!draggingRef.current) return
    updateFromClientX(event.clientX)
  }

  const handlePointerEnd = (event: PointerEvent<HTMLDivElement>) => {
    draggingRef.current = false
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 10 : 2

    switch (event.key) {
      case 'ArrowLeft':
        setPosition((current) => clamp(current - step))
        break
      case 'ArrowRight':
        setPosition((current) => clamp(current + step))
        break
      case 'Home':
        setPosition(0)
        break
      case 'End':
        setPosition(100)
        break
      default:
        return
    }

    event.preventDefault()
  }

  const rounded = Math.round(position)
  const objectFit = fit === 'contain' ? 'object-contain' : 'object-cover'

  return (
    <div
      className={`overflow-hidden rounded-2xl border border-ink-700 bg-ink-900 ${className}`}
    >
      <div
        ref={regionRef}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerEnd}
        onPointerCancel={handlePointerEnd}
        onPointerLeave={handlePointerEnd}
        style={aspectRatio ? { aspectRatio: String(aspectRatio) } : undefined}
        className={`relative w-full touch-pan-y select-none bg-ink-950 ${
          aspectRatio ? '' : 'aspect-[4/3]'
        }`}
      >
        <img
          src={beforeSrc}
          alt={`${beforeLabel} image`}
          draggable={false}
          className={`pointer-events-none absolute inset-0 h-full w-full ${objectFit}`}
        />
        <img
          src={afterSrc}
          alt={`${afterLabel} image`}
          draggable={false}
          style={{ clipPath: `inset(0 0 0 ${position}%)` }}
          className={`pointer-events-none absolute inset-0 h-full w-full ${objectFit}`}
        />

        <span className="pointer-events-none absolute top-3 left-3 rounded-full bg-ink-950/75 px-2.5 py-1 text-[11px] font-medium tracking-[0.16em] text-mist-200 uppercase backdrop-blur-sm">
          {beforeLabel}
        </span>
        <span className="pointer-events-none absolute top-3 right-3 rounded-full bg-ink-950/75 px-2.5 py-1 text-[11px] font-medium tracking-[0.16em] text-mist-200 uppercase backdrop-blur-sm">
          {afterLabel}
        </span>

        <div
          aria-hidden="true"
          style={{ left: `${position}%` }}
          className="pointer-events-none absolute inset-y-0 w-px bg-mist-50/70"
        />

        <div
          role="slider"
          tabIndex={0}
          aria-label="Before and after comparison position"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={rounded}
          aria-valuetext={`${rounded}% of the original image shown`}
          onKeyDown={handleKeyDown}
          style={{ left: `${position}%` }}
          className="absolute top-1/2 flex h-11 w-11 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize items-center justify-center rounded-full border border-mist-50/25 bg-ink-950/70 text-mist-50 outline-none backdrop-blur-sm transition-colors focus-visible:border-accent focus-visible:ring-2 focus-visible:ring-accent/50"
        >
          <svg
            viewBox="0 0 24 24"
            aria-hidden="true"
            className="h-4 w-4"
            fill="none"
            stroke="currentColor"
            strokeWidth={1.8}
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M9 6 4 12l5 6" />
            <path d="M15 6l5 6-5 6" />
          </svg>
        </div>
      </div>
    </div>
  )
}
