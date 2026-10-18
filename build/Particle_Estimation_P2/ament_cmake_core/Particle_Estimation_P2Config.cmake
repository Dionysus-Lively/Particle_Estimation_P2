# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_Particle_Estimation_P2_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED Particle_Estimation_P2_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(Particle_Estimation_P2_FOUND FALSE)
  elseif(NOT Particle_Estimation_P2_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(Particle_Estimation_P2_FOUND FALSE)
  endif()
  return()
endif()
set(_Particle_Estimation_P2_CONFIG_INCLUDED TRUE)

# output package information
if(NOT Particle_Estimation_P2_FIND_QUIETLY)
  message(STATUS "Found Particle_Estimation_P2: 0.0.0 (${Particle_Estimation_P2_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'Particle_Estimation_P2' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT Particle_Estimation_P2_DEPRECATED_QUIET)
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(Particle_Estimation_P2_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${Particle_Estimation_P2_DIR}/${_extra}")
endforeach()
