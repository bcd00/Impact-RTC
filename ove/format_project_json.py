import math

from utilities.utils import write_json, read_json


# noinspection DuplicatedCode
def main():
    xs = []
    section_width = 1920
    section_height = 1080

    data = read_json(f'./skeleton.json')
    sections = read_json(f'./sections.json')
    html_sections = read_json(f'./html_sections.json')

    for i in range(64):
        if i >= len(sections):
            continue

        section = sections[i]
        html = html_sections.get(section, None)

        if section == 'model_params':
            html = html_sections['model_params']
            xs.append({
                'space': 'DOCluster',
                'x': ((math.floor(i / 8) * 2) + (i % 2)) * section_width,
                'y': math.floor((i % 8) / 2) * section_height + html['offset'],
                'w': section_width / 2,
                'h': section_height,
                'app': {
                    'url': 'OVE_APP_HTML',
                    'states': {
                        'load': {
                            'url': html['url'].replace('%', str(html['version']))
                        }
                    }
                }
            })
            html = html_sections['model_results']
            xs.append({
                'space': 'DOCluster',
                'x': ((math.floor(i / 8) * 2) + (i % 2)) * section_width + (section_width / 2),
                'y': math.floor((i % 8) / 2) * section_height + html['offset'],
                'w': section_width / 2,
                'h': section_height,
                'app': {
                    'url': 'OVE_APP_HTML',
                    'states': {
                        'load': {
                            'url': html['url'].replace('%', str(html['version']))
                        }
                    }
                }
            })
            continue
        elif section == 'ablation_results':
            html = html_sections['ablation_results']
            xs.append({
                'space': 'DOCluster',
                'x': ((math.floor(i / 8) * 2) + (i % 2)) * section_width,
                'y': math.floor((i % 8) / 2) * section_height + html['offset'],
                'w': section_width / 2,
                'h': section_height,
                'app': {
                    'url': 'OVE_APP_HTML',
                    'states': {
                        'load': {
                            'url': html['url'].replace('%', str(html['version']))
                        }
                    }
                }
            })
            html = html_sections['ablation_dist']
            xs.append({
                'space': 'DOCluster',
                'x': ((math.floor(i / 8) * 2) + (i % 2)) * section_width + (section_width / 2),
                'y': math.floor((i % 8) / 2) * section_height + html['offset'],
                'w': section_width / 2,
                'h': section_height,
                'app': {
                    'url': 'OVE_APP_HTML',
                    'states': {
                        'load': {
                            'url': html['url'].replace('%', str(html['version']))
                        }
                    }
                }
            })
            continue
        if html is None:
            continue
        if html['skip']:
            xs.append({
                'space': 'DOCluster',
                'x': ((math.floor(i / 8) * 2) + (i % 2)) * section_width,
                'y': math.floor((i % 8) / 2) * section_height + html['offset'],
                'w': section_width,
                'h': section_height,
                'app': {
                    'url': 'OVE_APP_HTML',
                    'states': {
                        'load': {
                            'url': html['url']
                        }
                    }
                }
            })
        else:
            xs.append({
                'space': 'DOCluster',
                'x': ((math.floor(i / 8) * 2) + (i % 2)) * section_width,
                'y': math.floor((i % 8) / 2) * section_height + html['offset'],
                'w': section_width,
                'h': section_height - html['offset'],
                'app': {
                    'url': 'OVE_APP_HTML',
                    'states': {
                        'load': {
                            'url': html['url'].replace('%', str(html['version']))
                        }
                    }
                }
            })

    data['Sections'] = data['Sections'] + xs

    write_json(data, '../project.json')


if __name__ == '__main__':
    main()
