import { useRef, useState } from 'react'
import type { ChangeEvent, DragEvent, KeyboardEvent } from 'react'
import { formatBytes, formatExtensions } from '../lib/format'

export type UploadDropzoneProps = {
  onSelect: (file: File) => void
  acceptedExtensions: string[]
  maxSizeBytes: number
  busy?: boolean
}

/**
 * Touch-friendly image picker: click, tap or drop a file.
 * Format and size checks are repeated client-side for fast feedback; the
 * backend remains the source of truth.
 */
export function UploadDropzone({
  onSelect,
  acceptedExtensions,
  maxSizeBytes,
  busy = false,
}: UploadDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [isDragging, setIsDragging] = useState(false)

  const openPicker = () => {
    if (!busy) inputRef.current?.click()
  }

  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) onSelect(file)
    // Allow re-selecting the same file after a rejection.
    event.target.value = ''
  }

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragging(false)
    if (busy) return

    const file = event.dataTransfer.files?.[0]
    if (file) onSelect(file)
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      openPicker()
    }
  }

  return (
    <div
      role="button"
      tabIndex={busy ? -1 : 0}
      aria-label="Upload a low-light image"
      aria-busy={busy}
      aria-disabled={busy}
      onClick={openPicker}
      onKeyDown={handleKeyDown}
      onDragOver={(event) => {
        event.preventDefault()
        if (!busy) setIsDragging(true)
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={`flex min-h-64 cursor-pointer flex-col items-center justify-center gap-4 rounded-2xl border border-dashed px-5 py-10 text-center outline-none transition-colors focus-visible:ring-2 focus-visible:ring-accent/60 sm:min-h-72 ${
        isDragging
          ? 'border-accent/70 bg-ink-800'
          : 'border-ink-600 bg-ink-850 hover:border-ink-500 hover:bg-ink-800'
      } ${busy ? 'pointer-events-none opacity-70' : ''}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept={[
          ...acceptedExtensions,
          'image/jpeg',
          'image/png',
        ].join(',')}
        onChange={handleChange}
        className="sr-only"
        tabIndex={-1}
      />

      <span
        aria-hidden="true"
        className="flex h-12 w-12 items-center justify-center rounded-full border border-ink-600 bg-ink-900 text-accent"
      >
        <svg
          viewBox="0 0 24 24"
          className="h-5 w-5"
          fill="none"
          stroke="currentColor"
          strokeWidth={1.6}
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M12 16V4" />
          <path d="m7 9 5-5 5 5" />
          <path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
        </svg>
      </span>

      <div className="space-y-1">
        <p className="text-base font-medium text-mist-50">
          {busy ? 'Validating image…' : 'Drop a low-light photo here'}
        </p>
        <p className="text-sm text-mist-400">
          or tap to browse your device
        </p>
      </div>

      <p className="text-xs text-mist-500">
        {formatExtensions(acceptedExtensions)} · up to {formatBytes(maxSizeBytes)}
      </p>
    </div>
  )
}
