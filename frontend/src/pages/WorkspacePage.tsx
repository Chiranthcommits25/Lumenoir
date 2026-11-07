import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ComparisonSlider } from '../components/ComparisonSlider'
import { ImageInfoCard } from '../components/ImageInfoCard'
import { IntermediateGallery } from '../components/IntermediateGallery'
import { MethodSelector } from '../components/MethodSelector'
import { UploadDropzone } from '../components/UploadDropzone'
import {
  ApiError,
  enhanceImage,
  fetchPipelineStatus,
  fetchUploadLimits,
  uploadImage,
} from '../lib/api'
import type {
  EnhancementMode,
  EnhancementResult,
  ImageInfo,
  PipelineStatus,
  UploadLimits,
} from '../lib/api'
import { downloadBase64, downloadName, downloadObjectUrl } from '../lib/download'
import { formatBytes, formatExtensions } from '../lib/format'
import afterPlaceholder from '../assets/placeholder-after.svg'
import beforePlaceholder from '../assets/placeholder-before.svg'

type Status = 'idle' | 'validating' | 'processing'

// Mirrors the server defaults so the UI renders instantly; replaced by
// /api/images/limits as soon as the backend answers.
const FALLBACK_LIMITS: UploadLimits = {
  max_size_bytes: 10 * 1024 * 1024,
  allowed_extensions: ['.jpeg', '.jpg', '.png'],
  allowed_mime_types: ['image/jpeg', 'image/png'],
  dimensions: {
    min_width: 64,
    min_height: 64,
    max_width: 8000,
    max_height: 8000,
    max_megapixels: 40,
  },
}

export function WorkspacePage() {
  const [limits, setLimits] = useState<UploadLimits>(FALLBACK_LIMITS)
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [info, setInfo] = useState<ImageInfo | null>(null)
  const [mode, setMode] = useState<EnhancementMode>('hybrid')
  const [status, setStatus] = useState<Status>('idle')
  const [error, setError] = useState<string | null>(null)
  const [pendingStages, setPendingStages] = useState<string[]>([])
  const [result, setResult] = useState<EnhancementResult | null>(null)
  const requestRef = useRef(0)
  const previewUrlRef = useRef<string | null>(null)

  useEffect(() => {
    let active = true
    fetchUploadLimits()
      .then((next) => {
        if (active) setLimits(next)
      })
      .catch(() => {
        // Keep the fallback limits; endpoints still validate server-side.
      })

    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    let active = true
    fetchPipelineStatus(mode)
      .then((next) => {
        if (active) setPipelineStatus(next)
      })
      .catch(() => {
        if (active) setPipelineStatus(null)
      })

    return () => {
      active = false
    }
  }, [mode])

  // Preview URLs are created when the selection changes, not in an effect, so
  // the browser never holds more than one object URL at a time.
  const showPreview = useCallback((next: File | null) => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current)
      previewUrlRef.current = null
    }

    const url = next ? URL.createObjectURL(next) : null
    previewUrlRef.current = url
    setPreviewUrl(url)
  }, [])

  useEffect(
    () => () => {
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current)
    },
    [],
  )

  const handleSelect = useCallback(
    async (selected: File) => {
      const extension = `.${selected.name.split('.').pop()?.toLowerCase() ?? ''}`

      if (!limits.allowed_extensions.includes(extension)) {
        setFile(null)
        showPreview(null)
        setInfo(null)
        setResult(null)
        setStatus('idle')
        setError(
          `Unsupported file type. Use ${formatExtensions(limits.allowed_extensions)}.`,
        )
        return
      }

      if (selected.size === 0) {
        setFile(null)
        showPreview(null)
        setInfo(null)
        setResult(null)
        setStatus('idle')
        setError('That file is empty.')
        return
      }

      if (selected.size > limits.max_size_bytes) {
        setFile(null)
        showPreview(null)
        setInfo(null)
        setResult(null)
        setStatus('idle')
        setError(
          `That file is ${formatBytes(selected.size)}. The limit is ${formatBytes(limits.max_size_bytes)}.`,
        )
        return
      }

      const requestId = requestRef.current + 1
      requestRef.current = requestId

      setFile(selected)
      showPreview(selected)
      setInfo(null)
      setResult(null)
      setError(null)
      setPendingStages([])
      setStatus('validating')

      try {
        const image = await uploadImage(selected)
        if (requestRef.current !== requestId) return
        setInfo(image)
        setStatus('idle')
      } catch (cause) {
        if (requestRef.current !== requestId) return
        setFile(null)
        showPreview(null)
        setStatus('idle')
        setError(
          cause instanceof ApiError
            ? cause.message
            : 'Upload failed. Check that the backend is running and try again.',
        )
      }
    },
    [limits, showPreview],
  )

  const handleEnhance = useCallback(async () => {
    if (!file) return

    setStatus('processing')
    setError(null)
    setPendingStages([])

    try {
      const next = await enhanceImage(file, mode)
      setResult(next)
    } catch (cause) {
      setResult(null)
      if (cause instanceof ApiError) {
        setError(cause.message)
        setPendingStages(
          cause.payload?.pipeline?.stages
            .filter((stage) => stage.status !== 'implemented')
            .map((stage) => stage.name) ?? [],
        )
      } else {
        setError('Processing failed unexpectedly. Please try again.')
      }
    } finally {
      setStatus('idle')
    }
  }, [file, mode])

  const handleReset = () => {
    requestRef.current += 1
    setFile(null)
    showPreview(null)
    setInfo(null)
    setResult(null)
    setStatus('idle')
    setError(null)
    setPendingStages([])
  }

  const handleDownloadOriginal = () => {
    if (!previewUrl || !file) return
    const extension = file.name.split('.').pop()?.toLowerCase() || 'png'
    downloadObjectUrl(previewUrl, downloadName(file.name, 'original', extension))
  }

  const handleDownloadEnhanced = () => {
    if (!result) return
    downloadBase64(
      result.output.data,
      result.output.media_type,
      result.output.filename,
    )
  }

  const busy = status === 'validating' || status === 'processing'
  const activeMode = result?.pipeline.mode ?? mode
  const artifacts = useMemo(() => {
    const source = result?.pipeline.artifacts ?? pipelineStatus?.artifacts ?? []
    const modeStages =
      (pipelineStatus?.modes ?? []).find((item) => item.mode === activeMode)
        ?.required_stages ?? null
    if (!modeStages) return source
    const required = new Set(modeStages)
    return source.map((artifact) =>
      artifact.available || !artifact.produced_by
        ? artifact
        : required.has(artifact.produced_by)
          ? artifact
          : {
              ...artifact,
              applicable: false,
              reason: 'Not part of the selected enhancement mode.',
            },
    )
  }, [activeMode, mode, pipelineStatus, result])
  const modes = pipelineStatus?.modes ?? []

  return (
    <div className="mx-auto w-full max-w-6xl px-4 pb-16 sm:px-6 lg:px-8">
      <div className="flex flex-col gap-1 py-8 sm:py-10">
        <h1 className="text-2xl font-medium tracking-tight text-mist-50 sm:text-3xl">
          Workspace
        </h1>
        <p className="max-w-2xl text-sm leading-relaxed text-mist-400 sm:text-base">
          Upload a low-light photograph to inspect it and run it through the
          pipeline. Enhancement stages are reported honestly while the algorithms
          are still being researched.
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_20rem] lg:gap-6">
        <div className="flex min-w-0 flex-col gap-4">
          {error ? (
            <div
              role="alert"
              className="flex items-start gap-3 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200"
            >
              <svg
                aria-hidden="true"
                viewBox="0 0 24 24"
                className="mt-0.5 h-4 w-4 shrink-0"
                fill="none"
                stroke="currentColor"
                strokeWidth={1.8}
                strokeLinecap="round"
              >
                <circle cx="12" cy="12" r="9" />
                <path d="M12 8v5M12 16h.01" />
              </svg>
              <div className="min-w-0 space-y-1">
                <p className="break-words">{error}</p>
                {pendingStages.length ? (
                  <p className="text-xs text-red-300/80">
                    Pending stages: {pendingStages.join(', ')}
                  </p>
                ) : null}
              </div>
              <button
                type="button"
                onClick={() => setError(null)}
                className="ml-auto shrink-0 rounded text-xs text-red-300/80 underline-offset-4 outline-none hover:underline focus-visible:ring-2 focus-visible:ring-red-300/60"
              >
                Dismiss
              </button>
            </div>
          ) : null}

          {file ? (
            <section className="overflow-hidden rounded-2xl border border-ink-700 bg-ink-850">
              <div className="flex items-center justify-between gap-3 border-b border-ink-700 px-4 py-3">
                <p className="min-w-0 truncate text-sm text-mist-200" title={file.name}>
                  {file.name}
                </p>
                <button
                  type="button"
                  onClick={handleReset}
                  className="shrink-0 rounded-full border border-ink-600 px-3 py-1.5 text-xs text-mist-200 transition-colors outline-none hover:border-ink-500 hover:text-mist-50 focus-visible:ring-2 focus-visible:ring-accent/60"
                >
                  Replace
                </button>
              </div>

              <div className="flex items-center justify-center bg-ink-900 p-2 sm:p-4">
                {previewUrl ? (
                  <img
                    src={previewUrl}
                    alt={`Original preview of ${file.name}`}
                    className="max-h-[60vh] w-auto max-w-full rounded-lg object-contain"
                  />
                ) : null}
              </div>

              <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-ink-700 px-4 py-3 text-xs text-mist-400">
                <span
                  aria-hidden="true"
                  className={`h-1.5 w-1.5 rounded-full ${
                    busy ? 'animate-pulse bg-accent' : 'bg-emerald-400'
                  }`}
                />
                <span>
                  {status === 'processing'
                    ? 'Running the pipeline…'
                    : busy
                      ? 'Validating image…'
                      : 'Validated by the LUMENOIR API'}
                </span>
                <span className="text-ink-600">·</span>
                <span>{formatBytes(file.size)}</span>
              </div>
            </section>
          ) : (
            <UploadDropzone
              onSelect={handleSelect}
              acceptedExtensions={limits.allowed_extensions}
              maxSizeBytes={limits.max_size_bytes}
              busy={busy}
            />
          )}

          <section className="rounded-2xl border border-ink-700 bg-ink-850 p-4 sm:p-5">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
              <div className="min-w-0">
                <h2 className="text-sm font-medium text-mist-50">Enhancement</h2>
                <p className="mt-1 text-sm leading-relaxed text-mist-400">
                  Choose a method and run the pipeline. While stages are pending
                  the API returns the exact list rather than an invented image.
                </p>
              </div>
              <span className="w-fit shrink-0 rounded-full border border-ink-600 px-2.5 py-1 text-[10px] tracking-[0.14em] text-mist-500 uppercase">
                {pipelineStatus?.ready ? 'Available' : 'Pending'}
              </span>
            </div>

            <div className="mt-4">
              <MethodSelector
                value={mode}
                modes={modes}
                onChange={setMode}
                disabled={busy}
              />
            </div>

            <div className="mt-4 flex flex-col gap-2 sm:flex-row">
              <button
                type="button"
                onClick={handleEnhance}
                disabled={!file || busy}
                className="inline-flex min-h-11 flex-1 items-center justify-center gap-2 rounded-full bg-mist-50 px-5 text-sm font-medium text-ink-950 transition-colors outline-none hover:bg-white focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-ink-850 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {status === 'processing' ? 'Processing…' : 'Run enhancement'}
              </button>
              <button
                type="button"
                onClick={handleDownloadEnhanced}
                disabled={!result}
                title={
                  result
                    ? 'Download the enhanced image'
                    : 'Available once the enhancement pipeline is implemented'
                }
                className="inline-flex min-h-11 items-center justify-center rounded-full border border-ink-600 px-5 text-sm text-mist-200 transition-colors outline-none hover:border-ink-500 hover:text-mist-50 focus-visible:ring-2 focus-visible:ring-accent/60 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Download enhanced
              </button>
            </div>

            {file ? (
              <button
                type="button"
                onClick={handleDownloadOriginal}
                className="mt-2 text-xs text-mist-500 underline-offset-4 outline-none hover:text-mist-200 hover:underline focus-visible:ring-2 focus-visible:ring-accent/60"
              >
                Download the original instead
              </button>
            ) : null}
          </section>

          <section className="rounded-2xl border border-ink-700 bg-ink-850 p-4 sm:p-5">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
              <div className="min-w-0">
                <h2 className="text-sm font-medium text-mist-50">
                  Before / after comparison
                </h2>
                <p className="mt-1 text-sm leading-relaxed text-mist-400">
                  {result
                    ? 'Drag the handle to compare the original with the processed image.'
                    : 'The comparison uses a placeholder pair until a real enhancement result exists.'}
                </p>
              </div>
              <span className="w-fit shrink-0 rounded-full border border-ink-600 px-2.5 py-1 text-[10px] tracking-[0.14em] text-mist-500 uppercase">
                {result ? 'Real output' : 'Placeholder'}
              </span>
            </div>

            <ComparisonSlider
              className="mt-4"
              beforeSrc={result && previewUrl ? previewUrl : beforePlaceholder}
              afterSrc={
                result
                  ? `data:${result.output.media_type};base64,${result.output.data}`
                  : afterPlaceholder
              }
              beforeLabel="Original"
              afterLabel={result ? 'Enhanced' : 'Placeholder'}
              aspectRatio={result ? result.image.aspect_ratio : undefined}
              fit={result ? 'contain' : 'cover'}
            />

            <p className="mt-3 text-xs text-mist-500">
              Drag the handle, tap anywhere on the image, or focus it and use the
              arrow keys.
            </p>
          </section>

          <section className="rounded-2xl border border-ink-700 bg-ink-850 p-4 sm:p-5">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
              <div className="min-w-0">
                <h2 className="text-sm font-medium text-mist-50">
                  Intermediate outputs
                </h2>
                <p className="mt-1 text-sm leading-relaxed text-mist-400">
                  The explainable processing view. Each panel fills in only when
                  the backend actually produces that image.
                </p>
              </div>
            </div>

            <IntermediateGallery artifacts={artifacts} className="mt-4" />
          </section>
        </div>

        <aside className="flex min-w-0 flex-col gap-4">
          {info ? (
            <ImageInfoCard info={info} />
          ) : (
            <section className="rounded-2xl border border-ink-700 bg-ink-850 p-4 sm:p-5">
              <h2 className="text-sm font-medium text-mist-50">
                Awaiting an upload
              </h2>
              <p className="mt-2 text-sm leading-relaxed text-mist-400">
                Each upload is checked for file type, byte size, image
                dimensions and integrity before it is accepted.
              </p>
            </section>
          )}

          <section className="rounded-2xl border border-ink-700 bg-ink-850 p-4 sm:p-5">
            <h2 className="text-sm font-medium text-mist-50">Upload limits</h2>
            <dl className="mt-3 space-y-2 text-sm">
              <div className="flex items-baseline justify-between gap-4">
                <dt className="text-mist-500">Formats</dt>
                <dd className="text-right text-mist-200">
                  {formatExtensions(limits.allowed_extensions)}
                </dd>
              </div>
              <div className="flex items-baseline justify-between gap-4">
                <dt className="text-mist-500">Max size</dt>
                <dd className="text-right text-mist-200">
                  {formatBytes(limits.max_size_bytes)}
                </dd>
              </div>
              <div className="flex items-baseline justify-between gap-4">
                <dt className="text-mist-500">Dimensions</dt>
                <dd className="text-right text-mist-200">
                  {limits.dimensions.min_width}×{limits.dimensions.min_height} –{' '}
                  {limits.dimensions.max_width}×{limits.dimensions.max_height}
                </dd>
              </div>
              <div className="flex items-baseline justify-between gap-4">
                <dt className="text-mist-500">Max pixels</dt>
                <dd className="text-right text-mist-200">
                  {limits.dimensions.max_megapixels} MP
                </dd>
              </div>
            </dl>
          </section>

          <section className="rounded-2xl border border-ink-700 bg-ink-900 p-4 sm:p-5">
            <h2 className="text-sm font-medium text-mist-50">Inference</h2>
            <dl className="mt-3 space-y-2 text-sm">
              <div className="flex items-baseline justify-between gap-4">
                <dt className="text-mist-500">Model</dt>
                <dd className="text-right text-mist-200">
                  {pipelineStatus?.model.model_id || 'Not selected'}
                </dd>
              </div>
              <div className="flex items-baseline justify-between gap-4">
                <dt className="text-mist-500">State</dt>
                <dd className="text-right text-mist-200">
                  {pipelineStatus?.model.state ?? 'unknown'}
                </dd>
              </div>
              <div className="flex items-baseline justify-between gap-4">
                <dt className="text-mist-500">Device</dt>
                <dd className="text-right text-mist-200">
                  {pipelineStatus?.inference.device ?? 'cpu'}
                </dd>
              </div>
              <div className="flex items-baseline justify-between gap-4">
                <dt className="text-mist-500">Max edge</dt>
                <dd className="text-right text-mist-200">
                  {pipelineStatus?.inference.max_edge ?? 1600} px
                </dd>
              </div>
            </dl>
            <p className="mt-3 text-xs leading-relaxed text-mist-500">
              Architecture, weights and inference path are pending research. The
              loader, interface and CPU configuration are in place.
            </p>
          </section>
        </aside>
      </div>
    </div>
  )
}
