## Example script to generate datasets:

b2: both fluid and particle, smaller dataset:
```
python mpm256_ggui.py  --n_simu=500 --n_grid=128 --max_n_part_fluid=6000 --n_part_particle=200 --is_particle=True --is_save=True --is_gui=False 
```

f2: only fluid, smaller dataset:
```
python mpm256_ggui.py  --n_simu=500 --n_grid=128 --max_n_part_fluid=6000 --n_part_particle=200 --is_particle=False --is_save=True --is_gui=False
```

b: both fluid and particle, larger dataset:
```
python mpm256_ggui.py  --n_simu=500 --n_grid=128 --max_n_part_fluid=30000 --n_part_particle=1000 --is_particle=True --is_save=True --is_gui=False
```

f: only fluid, larger dataset:
```
python mpm256_ggui.py  --n_simu=500 --n_grid=128 --max_n_part_fluid=30000 --n_part_particle=1000 --is_particle=False --is_save=True --is_gui=False
```
