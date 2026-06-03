# all annotation is english
import time
import queue
import numpy as np
import sounddevice as sd
import torch
# import noisereduce as nr
from faster_whisper import WhisperModel
from .SpeechBuffer import SpeechBuffer

class AudioProcessor:
    def __init__(self, user_input_queue, interrupt_event, exit_event, **kwargs):
        self.user_input_queue = user_input_queue
        self.interrupt_event = interrupt_event
        self.exit_event = exit_event
        self.audio_data_queue = queue.Queue()

        # Create SpeechBuffer instance for 2-stage buffering
        self.speech_buffer = SpeechBuffer(silence_threshold_seconds=1.0, on_speech_end=self._on_final_sentence_ready)

        self.model = WhisperModel("large-v3", device="cuda", compute_type="float16")    # medium, large-v3

        # Silero VAD model + utilities loading
        vad_model, vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        # get_speech_timestamps, read_audio, VADIterator, collect_chunks
        (self.get_speech_timestamps, _, self.read_audio, self.VADIterator, self.collect_chunks) = vad_utils
        self.vad_model = vad_model

        # Configuration parameters with defaults
        self.RMS_THRESHOLD = kwargs.get("rms_threshold", 0.0015) # 0.001 ~ 0.00099 sensitivity
        self.SAMPLE_RATE = kwargs.get("sample_rate", 16000)      # AI recommended : 44100
        self.SILENCE_DURATION_FOR_PROCESSING = kwargs.get("silence_duration_for_processing", 0.8)
        self.VAD_THRESHOLD = kwargs.get("vad_threshold", 0.9)
        self.MIN_SPEECH_DURATION_MS = kwargs.get("min_speech_duration_ms", 250)
        self.MIN_SILENCE_DURATION_MS = kwargs.get("min_silence_duration_ms", 100)
        self.NOISE_GATE_THRESHOLD = kwargs.get("noise_gate_threshold", 0.007)
        self.COMPRESSOR_THRESHOLD = kwargs.get("compressor_threshold", 0.1)
        self.COMPRESSOR_RATIO = kwargs.get("compressor_ratio", 3.0)
        if self.COMPRESSOR_RATIO < 1.0:
            self.COMPRESSOR_RATIO = 1.0 # Ensure ratio is at least 1:1 (no compression)

        # Determine the frame size the VAD model's core component expects
        if self.SAMPLE_RATE == 16000:
            self.vad_model_expected_frame_size = 512
        elif self.SAMPLE_RATE == 8000: # Silero VAD also supports 8kHz
            self.vad_model_expected_frame_size = 256
        else:
            # Raise an error if the sample rate is not supported by the VAD model's typical configuration
            raise ValueError(f"AudioProcessing configured with SAMPLE_RATE={self.SAMPLE_RATE} Hz, but the Silero VAD model used expects 16000 Hz (for 512 sample chunks) or 8000 Hz (for 256 sample chunks).")

    def _on_final_sentence_ready(self, sentence: str):
        """Callback called when SpeechBuffer delivers the completed sentence."""
        if self.user_input_queue and not self.interrupt_event.is_set():
            print(f"\n[AudioProcessing] Final sentence assembled by SpeechBuffer: '{sentence}'")
            self.user_input_queue.put(sentence)

    def audio_callback(self, indata, frames, time_info, status):
        """Sounddevice InputStream callback: Save microphone chunk to user_input_queue."""
        if status:
            print(f"[BACKGROUND] sounddevice status: {status}")
        mono_chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        self.audio_data_queue.put(mono_chunk)

    def calculate_rms(self, audio_chunk):
        """RMS calculation"""
        return np.sqrt(np.mean(np.square(audio_chunk)))

    def _transcribe_audio_chunk(self, audio_chunk_np: np.ndarray):
        """
        Process the given audio chunk (NumPy array) with STT.
        Audio filtering has been disabled as it was causing speech distortion.
        """
        if audio_chunk_np.size == 0:
            return

        final_rms = self.calculate_rms(audio_chunk_np)
        if final_rms < self.RMS_THRESHOLD:
            return 

        try:
            segments, info = self.model.transcribe(
                audio_chunk_np, # Use the original NumPy array
                language="ko",
                beam_size=5,
            )
            text = "".join(seg.text for seg in segments).strip()

            if text:
                print(f"\n[AudioProcessing] Recognized fragment: '{text}' -> Sending to SpeechBuffer.")
                self.speech_buffer.add_transcript_fragment(text)
        except Exception as e:
            print(f"[AudioProcessing Error] Error during transcription: {e}")

    def _process_collected_buffer(self, collected_chunks: list):
        """
        [Refactoring] 중복되는 버퍼 처리 로직을 별도 메서드로 추출 (DRY 원칙 적용)
        수집된 오디오 청크들을 병합하고, 최소 길이를 만족하면 STT를 수행합니다.
        """
        if not collected_chunks:
            return

        concatenated_buffer = np.concatenate(collected_chunks).flatten()
        buffer_len_sec = len(concatenated_buffer) / self.SAMPLE_RATE
        
        # 로깅은 필요에 따라 조절
        # print(f"[VAD] Processing speech buffer. Length: {buffer_len_sec:.2f}s")

        if buffer_len_sec * 1000 >= self.MIN_SPEECH_DURATION_MS:
            self._transcribe_audio_chunk(concatenated_buffer)
        else:
            print(f"[VAD] Speech segment too short ({buffer_len_sec * 1000:.0f}ms), discarding.")
        
        collected_chunks.clear()

    def start_listening(self):
        collected_audio_chunks = []
        last_speech_time = time.time()
        is_currently_speaking = False
        current_segment_start_time = None # 현재 음성 구간의 시작 시간을 저장할 변수

        # VADIterator 인스턴스 생성
        vad_iterator = self.VADIterator(self.vad_model, threshold=self.VAD_THRESHOLD, 
                                        sampling_rate=self.SAMPLE_RATE, 
                                        min_silence_duration_ms=self.MIN_SILENCE_DURATION_MS, 
                                        speech_pad_ms=200)

        with sd.InputStream(channels=1, samplerate=self.SAMPLE_RATE, 
                            callback=self.audio_callback, blocksize=self.vad_model_expected_frame_size): # VAD 모델이 기대하는 프레임 크기로 변경
            print("\n[BACKGROUND] Listening using VAD...")

            while not self.exit_event.is_set():
                if self.exit_event.is_set():
                    break
                try:
                    # audio_data_queue에서 작은 오디오 조각(콜백에서 넣은 것)을 가져옴
                    audio_frame = self.audio_data_queue.get(timeout=0.1) # 짧은 타임아웃
                    # VADIterator에 오디오 프레임 제공
                    speech_dict = vad_iterator(audio_frame, return_seconds=True)

                    # 1. 발화 시작 감지
                    if speech_dict and "start" in speech_dict:
                        current_segment_start_time = speech_dict['start'] # 시작 시간 저장
                        print(f"[VAD] Speech start detected at {current_segment_start_time:.2f}s")

                        # 이미 수집된 버퍼가 있다면(이전 발화의 잔여물 등) 처리
                        if not is_currently_speaking and collected_audio_chunks:
                            print(f"[VAD] Processing previous speech buffer (due to new start).")
                            self._process_collected_buffer(collected_audio_chunks)
                        
                        is_currently_speaking = True

                    # 오디오 프레임 수집
                    if is_currently_speaking:
                        collected_audio_chunks.append(audio_frame)
                        last_speech_time = time.time()

                    # 3. 발화 종료 조건 확인 (VAD End 또는 침묵 타임아웃)
                    speech_ended_by_vad = speech_dict and "end" in speech_dict and is_currently_speaking
                    speech_ended_by_silence = is_currently_speaking and (time.time() - last_speech_time > self.SILENCE_DURATION_FOR_PROCESSING)

                    if speech_ended_by_vad or speech_ended_by_silence:
                        if speech_ended_by_vad:
                            print(f"[VAD] Speech end detected by VAD.")
                        elif speech_ended_by_silence:
                            print(f"[VAD] Silence duration exceeded.")

                        # [Refactoring] 공통 처리 메서드 호출
                        self._process_collected_buffer(collected_audio_chunks)
                        
                        is_currently_speaking = False
                        vad_iterator.reset_states() # VAD 상태 초기화
                        current_segment_start_time = None # 현재 구간 시작 시간 초기화
                    
                    # VAD가 'end'를 반환했지만 is_currently_speaking이 False인 경우 (예: 중복 'end' 또는 오류)
                    elif speech_dict and "end" in speech_dict and not is_currently_speaking:
                        vad_iterator.reset_states() # VAD 상태 초기화
                        collected_audio_chunks.clear() # 버퍼도 비워줌

                    # [ADDED] 루프마다 SpeechBuffer를 확인하여 텍스트 조각들 사이의 침묵을 감지합니다.
                    self.speech_buffer.check_for_speech_end()

                except queue.Empty:
                    # 큐가 비었을 때도 침묵 타임아웃 체크
                    if is_currently_speaking and (time.time() - last_speech_time > self.SILENCE_DURATION_FOR_PROCESSING):
                        print(f"[VAD] Silence duration exceeded during queue empty.")
                        self._process_collected_buffer(collected_audio_chunks)
                        
                        is_currently_speaking = False
                        vad_iterator.reset_states()
                        current_segment_start_time = None
                    else:
                        self.speech_buffer.check_for_speech_end()
                    continue
                except Exception as e:
                    print(f"[AudioProcessing] Error in listening loop: {e}")
                    # 오류 발생 시 버퍼 및 상태 초기화
                    collected_audio_chunks.clear()
                    is_currently_speaking = False
                    vad_iterator.reset_states()
                    current_segment_start_time = None
                    time.sleep(0.1) # 짧은 대기 후 계속

            # 종료 시 잔여 버퍼 처리
            if collected_audio_chunks:
                print("[VAD] Processing remaining speech buffer at exit.")
                self._process_collected_buffer(collected_audio_chunks)
            
            self.speech_buffer.flush()
            vad_iterator.reset_states()
            print("[BACKGROUND] Listening stopped.")


class AudioPlayback: 
    """
    Handles audio playback in a separate thread."""
    def __init__(self, audio_queue, interrupt_event, exit_event):
        self.audio_queue = audio_queue
        self.interrupt_event = interrupt_event
        self.exit_event = exit_event

    def _clear_audio_queue(self):
        """[Refactoring] 큐 비우기 로직 추출"""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break


    def audioplay_thread(self, sample_rate):
        stream = None
        try:
            stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype='float32')
            stream.start()
            print(f"[AudioPlayback] Playback thread started with samplerate {sample_rate}.")

            while not self.exit_event.is_set():
                if self.interrupt_event.is_set():
                    if stream.active: # 스트림이 활성화되어 재생 중이면 중지
                        stream.stop() # 현재 재생 중인 오디오 중단
                    self._clear_audio_queue() # [Refactoring] 메서드 사용
                    time.sleep(0.05) # 짧은 대기
                    continue

                try:
                    audio_fragment = self.audio_queue.get(timeout=0.1)
                    if audio_fragment is None: # 종료 신호일 수 있음 (현재 로직에서는 사용 안 함)
                        continue

                    if not stream.active and not self.interrupt_event.is_set(): # 인터럽트가 아닐 때만 스트림 재시작
                        stream.start()

                    if stream.active: # 스트림이 활성 상태일 때만 write
                        stream.write(audio_fragment)
                    self.audio_queue.task_done()

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[AudioPlayback] Error during playback: {e}")
                    # 오류 발생 시 스트림 재시작 시도
                    if stream:
                        stream.stop()
                        stream.close()
                    stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype='float32')
                    stream.start()


        except Exception as e:
            print(f"[AudioPlayback] Thread error: {e}")
        finally:
            if stream:
                stream.stop()
                stream.close()
            self._clear_audio_queue() # [Refactoring] 메서드 사용
            print("[AudioPlayback] Playback thread exited.")
        
