import argparse
import csv
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset_dir',
        help='Directory for dataset.',
        type=str,
    )
    parser.add_argument(
        'source_csv',
        help='Source CSV filename.',
        type=str,
    )
    parser.add_argument(
        '--column_data',
        nargs='+',
        help='Scrape columns and store with specified keys. Format: <meta_key>=<source_column_number>',
        required=True,
    )
    parser.add_argument(
        '--no_csv_header',
        help='CSV filename has NO header.',
        action='store_true',
    )
    args = parser.parse_args()
    if not os.path.isdir(args.dataset_dir):
        raise ValueError('Dataset directory "%s" is absent' % args.dataset_dir)

    columns = {}

    for cd in args.column_data:
        cd_parts = cd.split("=")
        if len(cd_parts) != 2:
            continue
        try:
            cd_parts[1] = int(cd_parts[1])
        except:
            continue
        columns[cd_parts[0]] = cd_parts[1]

    if len(columns) < 2:
        raise ValueError('Scrape columns not set or data format error. Should be at least 2 cols, one is for "name"')
    if 'name' not in columns:
        raise ValueError('"name" column must be set')

    try:
        with open(args.source_csv, mode='r') as source:
            csv_reader = csv.reader(source, delimiter=';')
            if not args.no_csv_header:
                next(csv_reader)
            rows, success = -1, 0
            for row in csv_reader:
                rows += 1
                name = _col_val(row, columns['name'])
                if name is None or name == '':
                    print("No name for position %d in row %s, skipped:" % (columns['name'], row))
                    continue
                dataset_class_dir = os.path.join(args.dataset_dir, name.replace(' ', '_'))
                if not os.path.isdir(dataset_class_dir):
                    print('Directory "%s" for "%s" is absent, skipped' % (dataset_class_dir, name))
                    continue
                data = {}
                for k in columns:
                    if k == 'name':
                        continue
                    val = _col_val(row, columns[k])
                    if val is not None and val != '':
                        try:
                            set_val = float(val)
                        except:
                            if '|' in val:
                                set_val = val.split('|')
                            else:
                                set_val = val
                        data[k] = set_val
                with open(os.path.join(dataset_class_dir, 'meta.json'), 'w') as meta_file:
                    json.dump(data, meta_file)
                print('Saved meta for "%s"' % name)
                success += 1
        print("Saved meta for %d persons from %d rows" % (success, rows))
    except Exception as e:
        print("Unable to read file %s: %s" % (args.source_csv, e))


def _col_val(row, index):
    if index < len(row):
        return row[index]
    return None


if __name__ == '__main__':
    main()
