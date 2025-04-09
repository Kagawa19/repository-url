import inspect
import logging
import traceback
import time
from functools import wraps

from ai_services_api.services.message.core.db_pool import get_pooled_connection, return_connection

logger = logging.getLogger(__name__)

# Dictionary to track which functions are creating connections
function_connection_counts = {}

def track_connections(func):
    """
    Decorator to track database connections created within a function.
    
    Usage:
        @track_connections
        def my_function():
            conn = get_pooled_connection()
            # ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        
        # Initialize count if not already present
        if func_name not in function_connection_counts:
            function_connection_counts[func_name] = {
                "created": 0,
                "returned": 0,
                "last_call": 0
            }
        
        # Record this call
        function_connection_counts[func_name]["last_call"] = time.time()
        
        # Get the original get_pooled_connection function
        original_get_pooled = globals().get('get_pooled_connection')
        
        # Wrap it to count connections
        def count_connections(*c_args, **c_kwargs):
            result = original_get_pooled(*c_args, **c_kwargs)
            function_connection_counts[func_name]["created"] += 1
            logger.debug(f"Function {func_name} created connection #{function_connection_counts[func_name]['created']}")
            return result
        
        # Replace the global function temporarily
        globals()['get_pooled_connection'] = count_connections
        
        # Get the original return_connection function
        original_return = globals().get('return_connection')
        
        # Wrap it to count returns
        def count_returns(*r_args, **r_kwargs):
            result = original_return(*r_args, **r_kwargs)
            function_connection_counts[func_name]["returned"] += 1
            logger.debug(f"Function {func_name} returned connection #{function_connection_counts[func_name]['returned']}")
            return result
        
        # Replace the global function temporarily
        globals()['return_connection'] = count_returns
        
        try:
            # Call the original function
            return func(*args, **kwargs)
        finally:
            # Restore original functions
            globals()['get_pooled_connection'] = original_get_pooled
            globals()['return_connection'] = original_return
            
            # Log connection balance
            created = function_connection_counts[func_name]["created"]
            returned = function_connection_counts[func_name]["returned"]
            
            if created != returned:
                logger.warning(f"Connection imbalance in {func_name}: created {created}, returned {returned}")
                # Log the stack to help identify where connections are being created
                logger.warning(f"Current stack:\n{traceback.format_stack()}")
            else:
                logger.debug(f"Connection balance in {func_name}: {created} created/returned")
    
    return wrapper

def diagnose_connection_issues():
    """
    Log current connection tracking information to help diagnose issues.
    
    Call this function periodically or when connection issues occur.
    """
    logger.info("----- Connection Usage Diagnosis -----")
    
    # Print information about connection tracking
    logger.info(f"Tracking {len(function_connection_counts)} functions that use connections")
    
    for func_name, counts in function_connection_counts.items():
        created = counts["created"]
        returned = counts["returned"]
        imbalance = created - returned
        last_call = counts["last_call"]
        time_since = time.time() - last_call if last_call else "never"
        
        status = "BALANCED" if imbalance == 0 else f"LEAKING ({imbalance} connections)"
        
        logger.info(f"Function {func_name}: {status}")
        logger.info(f"  - Created: {created}, Returned: {returned}")
        logger.info(f"  - Last called: {time_since:.1f} seconds ago")
    
    # Log pool status if available
    try:
        from ai_services_api.services.message.core.db_pool import log_pool_status
        log_pool_status()
    except ImportError:
        logger.info("Pool status unavailable - log_pool_status not imported")
    
    logger.info("--------------------------------------")

# Example usage
@track_connections
def example_leaky_function():
    """Example function that creates a connection but doesn't return it."""
    conn, pool, using_pool, conn_id = get_pooled_connection()
    # Oops, forget to return the connection!
    return "Result"

@track_connections
def example_proper_function():
    """Example function that properly closes its connection."""
    conn, pool, using_pool, conn_id = get_pooled_connection()
    try:
        return "Result"
    finally:
        return_connection(conn, pool, using_pool, conn_id)