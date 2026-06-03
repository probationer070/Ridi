import json
from redis import Redis
from typing import List, Optional

class RedisControl:
    """Base class for Redis data controllers."""
    def __init__(self, redis_client: Redis, redis_key: str):
        if not isinstance(redis_client, Redis):
            raise TypeError("redis_client must be a valid Redis client instance.")
        self.redis_client = redis_client
        self.redis_key = redis_key

    def clear_all(self) -> int:
        """Deletes the key from Redis."""
        return self.redis_client.delete(self.redis_key)

    def exists(self) -> bool:
        """Checks if the key exists in Redis."""
        return self.redis_client.exists(self.redis_key) > 0

class RedisListControl(RedisControl):
    """A generic controller for the Redis LIST data structure."""

    def get_all(self) -> List[str]:
        """Retrieves all items from the list."""
        return self.redis_client.lrange(self.redis_key, 0, -1)

    def get_range(self, start: int, end: int) -> List[str]:
        """Retrieves a range of items from the list."""
        return self.redis_client.lrange(self.redis_key, start, end)
    
    def get_len(self) -> int:
        """Returns the length of the list."""
        return self.redis_client.llen(self.redis_key)

    def push(self, value: str, to_front: bool = False):
        """Pushes a value to the list (right/end by default)."""
        if to_front:
            self.redis_client.lpush(self.redis_key, value)
        else:
            self.redis_client.rpush(self.redis_key, value)

    def pop(self, from_front: bool = False) -> Optional[str]:
        """Pops a value from the list (right/end by default)."""
        if from_front:
            return self.redis_client.lpop(self.redis_key)
        else:
            return self.redis_client.rpop(self.redis_key)

    def trim(self, start: int, end: int):
        """Trims the list to the specified range."""
        self.redis_client.ltrim(self.redis_key, start, end)

class RedisStringControl(RedisControl):
    """A generic controller for the Redis STRING data structure."""
    
    def get(self) -> Optional[str]:
        """Gets the value of the key."""
        return self.redis_client.get(self.redis_key)

    def set(self, value: str):
        """Sets the value for the key."""
        self.redis_client.set(self.redis_key, value)