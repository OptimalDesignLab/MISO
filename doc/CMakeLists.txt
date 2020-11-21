find_package(Doxygen)

if (DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CodeDocumentation.conf.in
    ${CMAKE_CURRENT_BINARY_DIR}/CodeDocumentation.conf @ONLY)

  if (UNIX)
    # Only create symlinks if UNIX operating system
    add_custom_target(doc
      COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/CodeDocumentation.conf
      COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/CodeDocumentation.html
      COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${CMAKE_CURRENT_BINARY_DIR}/CodeDocumentation/html/index.html
      ${CMAKE_CURRENT_BINARY_DIR}/CodeDocumentation.html
      BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/CodeDocumentation/html/index.html
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating API documentation with Doxygen to CodeDocumentation.html"
      VERBATIM)

    add_custom_target(clean-doc
      COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/CodeDocumentation.html
      COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/CodeDocumentation
      COMMENT "Removing API documentation"
      VERBATIM)

  else (UNIX)
    add_custom_target(doc
      COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/CodeDocumentation.conf
      BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/CodeDocumentation/html/index.html
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating API documentation with Doxygen to CodeDocumentation/html/index.html"
      VERBATIM)

    add_custom_target(clean-doc
      COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/CodeDocumentation
      COMMENT "Removing API documentation"
      VERBATIM)
  endif (UNIX)
endif (DOXYGEN_FOUND)
