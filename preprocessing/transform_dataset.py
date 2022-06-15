import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            bbx = member.find('bndbox')
            xmin = int(bbx.find('xmin').text)
            ymin = int(bbx.find('ymin').text)
            xmax = int(bbx.find('xmax').text)
            ymax = int(bbx.find('ymax').text)
            label = member.find('name').text

            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     label,
                     xmin,
                     ymin,
                     xmax,
                     ymax
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def convert_to_text_annotations(df, ds):
    with open(f"dataset/annotation_text/{ds}_annotation.txt", "w+") as f:
        for idx, row in df.iterrows():
            x1 = row['xmin']
            x2 = row['xmax']
            y1 = row['ymin']
            y2 = row['ymax']

            fileName = row['filename']
            className = row['class']
            f.write(fileName + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + className + '\n')
    print('Successfully converted csv to text file.')


def main():
    datasets = ['train', 'valid']
    for ds in datasets:
        image_path = os.path.join(os.getcwd(), 'annotations', ds)
        xml_df = xml_to_csv(image_path)
        xml_df['filename'] = xml_df['filename'].apply(lambda x: f'/images/{ds}/' + x)
        convert_to_text_annotations(xml_df, ds)
        xml_df.to_csv('dataset/csv_files/{}.csv'.format(ds), index=None)
        print('Successfully converted xml to csv.')


main()
