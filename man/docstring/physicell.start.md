# physicell.start()

## input:
```
    args 'path/to/setting.xml' file (string); default is 'config/PhysiCell_settings.xml'.

```

## output:
```
    physicell processing. 0 for success.

```

## run:
```python
    from embedding import physicell
    physicell.start('path/to/setting.xml')

```

## description:
```
    function (re)initializes physicell as specified in the settings.xml, cells.csv, and cell_rules.csv files and generates the step zero observation output. if run for re-initialization, it is assumed that start will not be called before stop has been called.
```