const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

export type EnhancementMode = 'classical' | 'deep_learning' | 'hybrid'

export type ChannelStatistics = {
  mean: number
  min: number
  max: number
  stddev: number
}

export type ImageStatistics = {
  luminance: ChannelStatistics
  red: ChannelStatistics
  green: ChannelStatistics
  blue: ChannelStatistics
  luminance_histogram: number[]
  histogram_bins: number
}

export type ImageInfo = {
  filename: string
  content_type: string
  format: string
  width: number
  height: number
  size_bytes: number
  colour_mode: string
  megapixels: number
  aspect_ratio: number
  statistics: ImageStatistics | null
}

export type UploadLimits = {
  max_size_bytes: number
  allowed_extensions: string[]
  allowed_mime_types: string[]
  dimensions: {
    min_width: number
    min_height: number
    max_width: number
    max_height: number
    max_megapixels: number
  }
}

export type StageStatus = 'implemented' | 'not_implemented'
export type ProcessingStatus = 'completed' | 'unavailable' | 'failed'

export type StageCapability = {
  name: string
  status: StageStatus
  description: string
}

export type ModeCapability = {
  mode: EnhancementMode
  label: string
  available: boolean
  required_stages: string[]
}

export type ModelStatus = {
  state: string
  model_id: string
  device: string
  weights_path: string | null
  message: string
  ready: boolean
  load_duration_ms: number | null
}

export type PipelineArtifact = {
  kind: string
  label: string
  description: string
  produced_by?: string | null
  available: boolean
  applicable?: boolean
  reason: string | null
  width: number | null
  height: number | null
  encoding: string | null
  media_type: string | null
  data: string | null
}

export type StageReport = {
  name: string
  status: StageStatus
  duration_ms: number | null
  detail: string | null
}

export type PipelineStatus = {
  ready: boolean
  stages: StageCapability[]
  modes: ModeCapability[]
  model: ModelStatus
  inference: {
    device: string
    max_edge: number
    max_inference_pixels: number
  }
  artifacts: PipelineArtifact[]
}

export type PipelineRun = {
  mode: EnhancementMode
  processing_status: ProcessingStatus
  processing_time_ms: number
  timings_ms: Record<string, number>
  stages: StageReport[]
  artifacts: PipelineArtifact[]
}

export type EnhancementOutput = {
  encoding: 'base64'
  media_type: string
  filename: string
  data: string
}

export type EnhancementResult = {
  ok: boolean
  mode: EnhancementMode
  processing_status: ProcessingStatus
  processing_time_ms: number
  image: ImageInfo
  pipeline: PipelineRun
  output: EnhancementOutput
}

export type MetricSpec = {
  name: string
  label: string
  status: string
  requires_reference: boolean
  requires_dependency: string | null
  description: string
}

type ErrorPayload = {
  error?: { code?: string; message?: string }
  pipeline?: PipelineRun
  model?: ModelStatus
  image?: ImageInfo
}

export class ApiError extends Error {
  code: string
  status: number
  payload: ErrorPayload | null

  constructor(
    message: string,
    code: string,
    status: number,
    payload: ErrorPayload | null = null,
  ) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.status = status
    this.payload = payload
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init)
  } catch {
    throw new ApiError(
      'Could not reach the LUMENOIR API. Check that the backend is running.',
      'network_error',
      0,
    )
  }

  const payload = (await response.json().catch(() => null)) as
    | ErrorPayload
    | null

  if (!response.ok) {
    throw new ApiError(
      payload?.error?.message ?? `Request failed (${response.status}).`,
      payload?.error?.code ?? 'request_failed',
      response.status,
      payload,
    )
  }

  return payload as T
}

/** Validate an image server-side and return its information and statistics. */
export async function uploadImage(file: File): Promise<ImageInfo> {
  const body = new FormData()
  body.append('file', file)

  const payload = await request<{ image: ImageInfo }>('/api/images/upload', {
    method: 'POST',
    body,
  })

  return payload.image
}

/** Read the server's upload constraints so the UI shows the real limits. */
export function fetchUploadLimits(): Promise<UploadLimits> {
  return request<UploadLimits>('/api/images/limits')
}

/** Read pipeline, model and mode availability. */
export function fetchPipelineStatus(mode?: EnhancementMode): Promise<PipelineStatus> {
  const query = mode ? `?mode=${encodeURIComponent(mode)}` : ''
  return request<PipelineStatus>(`/api/enhance/status${query}`)
}

/**
 * Run the enhancement pipeline.
 *
 * While the stages are unimplemented this rejects with
 * ``ApiError.code === 'enhancement_unavailable'`` and the pending-stage
 * report attached to ``payload.pipeline`` — no output image is invented.
 */
export function enhanceImage(
  file: File,
  mode: EnhancementMode,
): Promise<EnhancementResult> {
  const body = new FormData()
  body.append('file', file)
  body.append('mode', mode)

  return request<EnhancementResult>('/api/enhance', {
    method: 'POST',
    body,
  })
}

/** Declared evaluation metrics (none computed yet). */
export function fetchMetrics(): Promise<{
  status: string
  metrics: MetricSpec[]
}> {
  return request('/api/evaluation/metrics')
}
