In this folder a generator of the INP files of the BPDRR.
<br>
This is more suitable for the purposes of current EPANET 2.2 where 
- PDA is already implemented and there is no need to add new elements as in Paez et al. 2020
- simplifies the INP files
- include emitters. Interestingly WNTR takes the emitter coefficient but makes it into LPS. For that reason the value needs to be divided by 1000 when loaded
