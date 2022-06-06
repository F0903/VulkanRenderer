export module Validation;

#ifdef NDEBUG
export constinit bool ENABLE_VALIDATION_LAYERS = false;
#else
export constinit bool ENABLE_VALIDATION_LAYERS = true;
#endif // NDEBUG