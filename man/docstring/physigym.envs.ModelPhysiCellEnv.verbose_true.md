# physigym.envs.ModelPhysiCellEnv.verbose_true()


## input:
```

```

## output:
```

```python

## run:
```
            import gymnasium
            import physigym

            env = gymnasium.make('physigym/ModelPhysiCellEnv')

            env.verbose_true()

```

## description:
```
            run env.unwrapped.verbose_true()
            to set verbosity true after initialization.

            please not, only little from the standard output is coming
            actually from physigym. most of the output comes straight
            from PhysiCell and this setting has no influence over that output.
        
```