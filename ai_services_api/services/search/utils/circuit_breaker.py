import time
import logging
from typing import Callable, Any

class CircuitBreaker:
    """
    Implements circuit breaker pattern for external API calls.
    
    This prevents unnecessary calls to failing services and allows
    the system to recover gracefully.
    """
    
    def __init__(self, name, failure_threshold=5, recovery_timeout=30, 
                 retry_timeout=60):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name of the service protected by this circuit breaker
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Time in seconds to wait before attempting recovery
            retry_timeout: Time in seconds before resetting failure count
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.retry_timeout = retry_timeout
        
        self.failures = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = 0
        self.last_attempt_time = 0
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute the function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the function or None if circuit is open
        """
        current_time = time.time()
        
        # Check if we should reset failure count
        if current_time - self.last_failure_time > self.retry_timeout:
            if self.failures > 0:
                self.logger.info(f"Circuit breaker '{self.name}': Resetting failure count")
                self.failures = 0
        
        # Check circuit state
        if self.state == "OPEN":
            # Check if recovery timeout has elapsed
            if current_time - self.last_failure_time > self.recovery_timeout:
                self.logger.info(f"Circuit breaker '{self.name}': Transitioning to HALF-OPEN")
                self.state = "HALF-OPEN"
            else:
                self.logger.warning(f"Circuit breaker '{self.name}': Circuit OPEN, request rejected")
                return None
        
        # Try to execute the function
        try:
            self.last_attempt_time = current_time
            result = await func(*args, **kwargs)
            
            # If we got here in HALF-OPEN, close the circuit
            if self.state == "HALF-OPEN":
                self.logger.info(f"Circuit breaker '{self.name}': Closing circuit after successful execution")
                self.state = "CLOSED"
                self.failures = 0
            
            return result
            
        except Exception as e:
            # Record the failure
            self.failures += 1
            self.last_failure_time = current_time
            
            # Check if we should open the circuit
            if self.failures >= self.failure_threshold:
                if self.state != "OPEN":
                    self.logger.warning(
                        f"Circuit breaker '{self.name}': Opening circuit after {self.failures} failures"
                    )
                    self.state = "OPEN"
            
            # Re-raise the exception
            raise e