from configs import QueueManager, EventManager
from .AudioProcess import AudioProcessor, AudioPlayback

def background_listening_thread(exit_event):
    """백그라운드 음성 인식 스레드 함수"""
    qm = QueueManager.get_instance()
    em = EventManager.get_instance()
    stt = AudioProcessor(
        user_input_queue=qm.user_input_queue,
        interrupt_event=em.interrupt_event,
        exit_event=exit_event
    )
    stt.start_listening()
    print("[AudioThreads] background_listening_thread Exited cleanly")


def audio_playback_thread(sample_rate, exit_event):
    """오디오 재생 스레드 함수"""
    qm = QueueManager.get_instance()
    em = EventManager.get_instance()
    audio_playback = AudioPlayback(
        audio_queue=qm.audio_queue,
        interrupt_event=em.interrupt_event,
        exit_event=exit_event
    )
    audio_playback.audioplay_thread(sample_rate)
    print("[AudioThreads] audio_playback_thread Exited cleanly")
