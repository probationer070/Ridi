def run_sovits_tts(tts_pipeline, text, system_config):
    """
    GPT_SoVITS에서 제공하는 TTS Pipeline 호출
    - tts_pipeline: TTS 파이프라인 객체
    - text: 합성할 문장
    - ref_audio_path: 레퍼런스 음성 파일 경로
    - ref_text: 레퍼런스 음성에 대한 텍스트
    """
    inputs = {
        "text": text,
        "text_lang": "ko",  # 실제 언어에 맞게 수정
        "ref_audio_path": system_config.ref_audio_path,
        "prompt_text": system_config.ref_text,
        "prompt_lang": "en",  # 예시
        "top_k": 5,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": "cut0",
        "batch_size": 1,
        "batch_threshold": 0.75,
        "split_bucket": True,
        "speed_factor": 1.0,
        "fragment_interval": 0.1,
        "seed": 2509395972,
        "return_fragment": False,
        "parallel_infer": True,
        "repetition_penalty": 1.35,
    }
    synthesis_result = tts_pipeline.run_generator(inputs)
    return synthesis_result
