idf_component_register(SRCS "gattc_demo.c"
                    INCLUDE_DIRS ".")

spiffs_create_partition_image(storage ../spiffs_data FLASH_IN_PROJECT)
