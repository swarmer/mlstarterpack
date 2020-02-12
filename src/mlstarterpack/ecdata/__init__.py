r"""Entity Component Data
Utilities for working with datasets formatted as
what is called here "Entity Component Data".

# The format
The idea of the format is to split source data objects
(images, audio, texts or table rows) from associated data
such as class labels, object outlines etc. This is somewhat similar to the idea of
ECS architecture in game development.

To associate a source object with related data, each object must have a unique key:
usually it's the path of the file relative to the root data directory,
but for tabular data it can be a value of a column in the source data frame.

\<root data dir\>:
- datasets/
    - \<annotationname\>_\<splitname\>.csv: simply a one-columnt list of object
        keys belonging to the dataset
    - class_train.csv: as an example
    - \<other formats TBD\>
- processed/
    - Any data with preprocessing applied after taking source data
    - Usually this directory will mirror contents of the `source/` directory
- source/
    - \<subdataset\>/*: Source data as it's originally received
            (but possibly renamed / with a key added or changed).
        - images/0b3eacdd-408a-44f7-a42c-3c8004c189aa.jpg: as an example
        - It's recommended to use UUIDs for object keys/names to ensure they never
            intersect and datasets can be easily merged.
        - Files can be split into subdirectories to avoid having too many files
            in a single directory.
    - metadata/
        - <annotationname>.csv: (usually) two columns: object key and the annotation
            (usually as one column)
        - class.csv: as an example
"""
