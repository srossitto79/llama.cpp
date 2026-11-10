get_filename_component(DEST_DIR "${DEST}" DIRECTORY)
file(MAKE_DIRECTORY "${DEST_DIR}")

# URL is optional, the default is the ggml-org/models repository
if(NOT DEFINED URL)
    set(URL "https://huggingface.co/ggml-org/models/resolve/main/${NAME}?download=true")
endif()

if(NOT EXISTS "${DEST}")
    message(STATUS "Downloading ${NAME} from ${URL}...")
endif()

# an existing file with the expected hash is not downloaded again
file(DOWNLOAD
    "${URL}"
    "${DEST}"
    TLS_VERIFY ON
    EXPECTED_HASH ${HASH}
    STATUS status
)

list(GET status 0 code)

if(NOT code EQUAL 0)
    list(GET status 1 msg)
    message(FATAL_ERROR "Failed to download ${NAME}: ${msg}")
endif()
