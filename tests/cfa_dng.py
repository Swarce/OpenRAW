"""Synthetic Bayer (CFA) DNG builder for tests: known pattern, levels and content.
Reused groundwork for the planned CFA/Bayer output mode."""
import numpy as np, tifffile
PAT = {"RGGB": (0,1,1,2), "GRBG": (1,0,2,1), "BGGR": (2,1,1,0), "GBRG": (1,2,0,1)}
def make_cfa_dng(path, pattern="GRBG", h=96, w=144, black=256, white=4095, make="Testco", model="T1"):
    """Synthetic Bayer DNG: left third pure red, middle pure green, right pure blue."""
    rgb = np.zeros((h, w, 3)); third = w // 3
    rgb[:, :third, 0] = 0.8; rgb[:, third:2*third, 1] = 0.8; rgb[:, 2*third:, 2] = 0.8
    idx = np.array(PAT[pattern]).reshape(2, 2)
    chan = np.tile(idx, (h // 2, w // 2))
    mosaic = np.take_along_axis(rgb, chan[..., None], 2)[..., 0]
    data = (black + mosaic * (white - black)).astype(np.uint16)
    cm = [3240455,1000000,-1537139,1000000,-498532,1000000,-969266,1000000,1876011,1000000,41556,1000000,55643,1000000,-204026,1000000,1057225,1000000]
    tags = [(50706,'B',4,(1,4,0,0),False),(50707,'B',4,(1,1,0,0),False),(271,'s',0,make,False),(272,'s',0,model,False),
            (50708,'s',0,f"{make} {model}",False),(33421,'H',2,(2,2),False),(33422,'B',4,PAT[pattern],False),
            (50710,'B',3,(0,1,2),False),(50711,'H',1,1,False),(50714,'H',1,black,False),(50717,'H',1,white,False),
            (50778,'H',1,21,False),(50721,'2i',9,cm,False),(50728,'2I',3,[1,1,1,1,1,1],False)]
    tifffile.imwrite(path, data, photometric=32803, extratags=tags, metadata=None, subfiletype=0)
