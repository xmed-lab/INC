import json, os, cv2

names16 = ["normal_001", "normal_002", "normal_003", "normal_004", "normal_005", "normal_006", "normal_007", "normal_008", "normal_009", "normal_010", "normal_011", "normal_012", "normal_013", "normal_014", "normal_015", "normal_016", "normal_017", "normal_018", "normal_019", "normal_020", "normal_021", "normal_022", "normal_023", "normal_024", "normal_025", "normal_026", "normal_027", "normal_028", "normal_029", "normal_030", "normal_031", "normal_032", "normal_033", "normal_034", "normal_035", "normal_036", "normal_037", "normal_038", "normal_039", "normal_040", "normal_041", "normal_042", "normal_043", "normal_044", "normal_045", "normal_046", "normal_047", "normal_048", "normal_049", "normal_050", "normal_051", "normal_052", "normal_053", "normal_054", "normal_055", "normal_056", "normal_057", "normal_058", "normal_059", "normal_060", "normal_061", "normal_062", "normal_063", "normal_064", "normal_065", "normal_066", "normal_067", "normal_068", "normal_069", "normal_070", "normal_071", "normal_072", "normal_073", "normal_074", "normal_075", "normal_076", "normal_077", "normal_078", "normal_079", "normal_080", "normal_081", "normal_082", "normal_083", "normal_084", "normal_085", "normal_087", "normal_088", "normal_089", "normal_090", "normal_091", "normal_092", "normal_093", "normal_094", "normal_095", "normal_096", "normal_097", "normal_098", "normal_099", "normal_100", "normal_101", "normal_102", "normal_103", "normal_104", "normal_105", "normal_106", "normal_107", "normal_108", "normal_109", "normal_110", "normal_111", "normal_112", "normal_113", "normal_114", "normal_115", "normal_116", "normal_117", "normal_118", "normal_119", "normal_120", "normal_121", "normal_122", "normal_123", "normal_124", "normal_125", "normal_126", "normal_127", "normal_128", "normal_129", "normal_130", "normal_131", "normal_132", "normal_133", "normal_134", "normal_135", "normal_136", "normal_137", "normal_138", "normal_139", "normal_140", "normal_141", "normal_142", "normal_143", "normal_144", "normal_145", "normal_146", "normal_147", "normal_148", "normal_149", "normal_150", "normal_151", "normal_152", "normal_153", "normal_154", "normal_155", "normal_156", "normal_157", "normal_158", "normal_159", "normal_160", "test_001", "test_002", "test_003", "test_004", "test_005", "test_006", "test_007", "test_008", "test_009", "test_010", "test_011", "test_012", "test_013", "test_014", "test_015", "test_016", "test_017", "test_018", "test_019", "test_020", "test_021", "test_022", "test_023", "test_024", "test_025", "test_026", "test_027", "test_028", "test_029", "test_030", "test_031", "test_032", "test_033", "test_034", "test_035", "test_036", "test_037", "test_038", "test_039", "test_040", "test_041", "test_042", "test_043", "test_044", "test_045", "test_046", "test_047", "test_048", "test_050", "test_051", "test_052", "test_053", "test_054", "test_055", "test_056", "test_057", "test_058", "test_059", "test_060", "test_061", "test_062", "test_063", "test_064", "test_065", "test_066", "test_067", "test_068", "test_069", "test_070", "test_071", "test_072", "test_073", "test_074", "test_075", "test_076", "test_077", "test_078", "test_079", "test_080", "test_081", "test_082", "test_083", "test_084", "test_085", "test_086", "test_087", "test_088", "test_089", "test_090", "test_091", "test_092", "test_093", "test_094", "test_095", "test_096", "test_097", "test_098", "test_099", "test_100", "test_101", "test_102", "test_103", "test_104", "test_105", "test_106", "test_107", "test_108", "test_109", "test_110", "test_111", "test_112", "test_113", "test_114", "test_115", "test_116", "test_117", "test_118", "test_119", "test_120", "test_121", "test_122", "test_123", "test_124", "test_125", "test_126", "test_127", "test_128", "test_129", "test_130", "tumor_001", "tumor_002", "tumor_003", "tumor_004", "tumor_005", "tumor_006", "tumor_007", "tumor_008", "tumor_009", "tumor_010", "tumor_011", "tumor_012", "tumor_013", "tumor_014", "tumor_015", "tumor_016", "tumor_017", "tumor_018", "tumor_019", "tumor_020", "tumor_021", "tumor_022", "tumor_023", "tumor_024", "tumor_025", "tumor_026", "tumor_027", "tumor_028", "tumor_029", "tumor_030", "tumor_031", "tumor_032", "tumor_033", "tumor_034", "tumor_035", "tumor_036", "tumor_037", "tumor_038", "tumor_039", "tumor_040", "tumor_041", "tumor_042", "tumor_043", "tumor_044", "tumor_045", "tumor_046", "tumor_047", "tumor_048", "tumor_049", "tumor_050", "tumor_051", "tumor_052", "tumor_053", "tumor_054", "tumor_055", "tumor_056", "tumor_057", "tumor_058", "tumor_059", "tumor_060", "tumor_061", "tumor_062", "tumor_063", "tumor_064", "tumor_065", "tumor_066", "tumor_067", "tumor_068", "tumor_069", "tumor_070", "tumor_071", "tumor_072", "tumor_073", "tumor_074", "tumor_075", "tumor_076", "tumor_077", "tumor_078", "tumor_079", "tumor_080", "tumor_081", "tumor_082", "tumor_083", "tumor_084", "tumor_085", "tumor_086", "tumor_087", "tumor_088", "tumor_089", "tumor_090", "tumor_091", "tumor_092", "tumor_093", "tumor_094", "tumor_095", "tumor_096", "tumor_097", "tumor_098", "tumor_099", "tumor_100", "tumor_101", "tumor_102", "tumor_103", "tumor_104", "tumor_105", "tumor_106", "tumor_107", "tumor_108", "tumor_109", "tumor_110", "tumor_111"]

temp = []
for _ in names16:
    if 'test' not in _:
        temp.append(_)
names16 = temp

anno17 = {}
for _ in open('/home/ylini/CAMELYON17/stages.csv').readlines():
    if 'node' in _:
        k, lb, center = _.strip().split(',')
        lb = 0 if lb == 'negative' else 1
        anno17[k] = [lb, int(center)]

out_json = {}

gt_dir = '/home/ylini/CAMELYON16/patch/gt/'

for n in names16:
    out_json[n] = {}
    if n + '.png' in os.listdir(gt_dir):
        out_json[n]['patch_labels'] = gt_dir + n + '.png'
        out_json[n]['wsi_label'] = 1
        gt = cv2.imread(out_json[n]['patch_labels'])[:, :, 0]
        out_json[n]['pos_patch_num'] = int(gt.sum())
    else:
        out_json[n]['wsi_label'] = 0
    out_json[n]['fixed_test_set'] = False

for k, v in anno17.items():
    n = k.split('.')[0]
    out_json[n] = {}
    out_json[n]['wsi_label'] = v[0]
    out_json[n]['center'] = v[1]
    out_json[n]['fixed_test_set'] = True

outs = json.dumps(out_json, indent=4)
open('/home/ylini/open_wsi/dataset/camelyon17.json', 'w').write(outs)
