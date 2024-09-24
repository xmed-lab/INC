import os, sys, math
import multiprocessing


def dzsave(files, in_dir, out_dir):
    for f in files:
        os.system('vips dzsave ' + os.path.join(in_dir, f) + ' ' + os.path.join(out_dir, f[:-4]) + ' --tile-size 256')
        os.system('sleep 3s')

        root = os.path.join(out_dir, f[:-4] + '_files')
        ds = [int(_) for _ in os.listdir(root)]
        ds.sort()
        keep = ds[-2] # keep x20 only for camelyon and tcga [-1] is x40
        for d in ds:
            if d != keep:
                os.system('rm -rf ' + os.path.join(root, str(d)))

        if os.path.exists(root):
            print(os.listdir(root))
        else:
            print(root)


def main():

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    worker_num = int(sys.argv[3])
    
    os.makedirs(out_dir, exist_ok=True)
    files = os.listdir(in_dir)
    temp = []
    for f in files:
        if f.replace('.svs', '.dzi') not in os.listdir(out_dir):
            temp.append(f)
    files = temp

    num_each = math.ceil(len(files) / worker_num)
    worker_num = math.ceil(len(files) / num_each)

    pool = multiprocessing.Pool(processes = worker_num)
    for i in range(worker_num):
        start = i * num_each
        end = i * num_each + num_each
        if start < len(files):
            pool.apply_async(dzsave, (files[start: end], in_dir, out_dir))
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()

