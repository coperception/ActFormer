## V2X-Sim

Download v2X-Sim V2.0 full dataset data

**Prepare data**

*We genetate custom annotation files which are different from mmdet3d's*

```
python tools/create_data.py v2x_sim --root-path ./data/v2x_sim --out-dir ./data/v2x_sim --extra-tag v2x_sim --version v2.0-mini --canbus ./data
```

Using the above code will generate `v2xsim_infos_temporal_{train,val}.pkl`.
