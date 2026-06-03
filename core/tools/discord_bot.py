import discord
import asyncio
from typing import Dict, Optional

from multiprocessing.synchronize import Event as EventType
from utils2.debug_logger import setup_logger

class DiscordClient(discord.Client):
    def __init__(self, *, intents: discord.Intents, queues: Dict, exit_event: EventType, target_channel_id: Optional[int] = None, **options):
        super().__init__(intents=intents, **options)
        setup_logger(log_path="discord.log", mode='all')
        self.user_input = queues.get('user_input')
        self.response_queue = queues.get('discord_response_queue')
        self.exit_event = exit_event # 메인 프로세스로부터 exit_event를 직접 받음
        self.target_channel_id = target_channel_id
        self.response_task = None

    async def on_ready(self):
        print(f'[Discord Bot] Logged on as {self.user}')
        # 특정 채널 ID가 설정되었다면 해당 채널 객체를 가져옵니다.
        if self.target_channel_id:
            channel = self.get_channel(self.target_channel_id)
            if channel:
                await channel.send("Ridi 시스템이 온라인 상태입니다. 이제 대화를 시작할 수 있습니다.")
            else:
                print(f"[Discord Bot] Error: Channel with ID {self.target_channel_id} not found.")
        
        # AI 응답을 처리하는 백그라운드 작업을 시작합니다.
        self.response_task = asyncio.create_task(self.process_responses())

    async def on_message(self, message: discord.Message):
        # 봇 자신의 메시지는 무시합니다.
        if message.author == self.user:
            return

        # 1. 개인 메시지(DM) 처리
        if isinstance(message.channel, discord.DMChannel):
            content = message.content
            print(f'[Discord Bot] Received DM from {message.author}: {content}')
            # 전달받은 user_input에 직접 삽입
            self.user_input.put({'source': 'discord', 'content': content, 'author_id': message.author.id, 'author_name': message.author.name})
            # message의 내용 확인하는 코드 예시
            print(f'[Discord Bot] Message info: {message}')
        
        # 2. 그 외 채널 메시지 처리 (채널 ID 동적 처리)
        else:
            content = message.content
            print(f'[Discord Bot] Received message from channel {message.channel.id}: {content}')
            self.user_input.put({'source': 'discord', 'content': content, 'channel_id': message.channel.id})

    async def process_responses(self):
        """AI 응답 큐를 확인하고 Discord로 메시지를 보냅니다."""
        loop = asyncio.get_running_loop()
        while not self.exit_event.is_set():
            try:
                # 비동기 환경에서 queue.get()을 직접 호출하면 블로킹되므로,
                # run_in_executor를 사용하여 별도 스레드에서 실행합니다.
                response_data = await loop.run_in_executor(None, self.response_queue.get) # self.response_queue is already multiprocessing.Queue
                if response_data is None: # 종료 신호
                    break

                response_text = response_data.get('content')
                source_info = response_data.get('source_info', {})
                author_id = source_info.get('author_id')
                channel_id = source_info.get('channel_id')

                # 응답을 보낼 채널 결정
                # 1. DM으로 온 요청이면 해당 사용자에게 DM으로 응답
                if author_id:
                    user = self.get_user(author_id) or await self.fetch_user(author_id)
                    if user:
                        await user.send(response_text)
                # 2. 서버 채널에서 온 요청이면 해당 채널로 응답
                elif channel_id:
                    channel = self.get_channel(channel_id)
                    if not channel:
                        try:
                            channel = await self.fetch_channel(channel_id)
                        except Exception as e:
                            print(f"[Discord Bot] Failed to fetch channel {channel_id}: {e}")
                    if channel:
                        await channel.send(response_text)

                # multiprocessing.Queue에는 task_done()이 없으므로 호출하지 않습니다.
            except Exception as e:
                # 큐가 비어있을 때 발생하는 예외는 무시하고 계속 진행
                if not isinstance(e, asyncio.QueueEmpty):
                    print(f"[Discord Bot] Error processing response queue: {e}")
            
            if self.exit_event.is_set():
                break
        print("[Discord Bot] Response processing task finished.")
