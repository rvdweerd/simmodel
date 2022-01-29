from PIL import Image
# img = Image.open("z3x3orig.png")
# img2=img.crop((110,0,530,480))
# img2.save("c_z3x3orig.png")
# img = Image.open("z5x5orig.png")
# img2=img.crop((110,0,530,480))
# img2.save("c_z5x5orig.png")
# img = Image.open("zMetorig.png")
# img2=img.crop((80,0,560,480))
# img2.save("c_Metorig.png")

from pathlib import Path
import os
working_dir = Path()
for path in working_dir.glob("**/*.png"):
    #print(path)
    # OR if you need absolute paths
    pathname=str(path.absolute())
    #if 'Entry=' in pathname and 'c_' not in pathname:
    if 'example_3x3in' in pathname:
        dirname = os.path.dirname(pathname)
        fname = path.parts[-1]
        fstem = path.stem
        
        #if '3x3' in pathname or '5x5' in pathname:
        if 'example' in pathname:
            img = Image.open(pathname)
            img2=img.crop((110,0,530,480))
            img2.save(dirname+'/c_'+fname)
            os.remove(pathname)
        elif 'Metro' in pathname:
            img = Image.open(pathname)
            img2=img.crop((80,0,560,480))
            img2.save(dirname+'/c_'+fname)
            os.remove(pathname)
        else:
            print('Not processed',fname)




    # OR if you need only filenames without extension for further parsing
    #print(path.stem)