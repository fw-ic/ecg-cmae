def parse_headers(fl_pth, prnt_opt=False):

    ecg_metadata = {}

    fp = open(fl_pth,'r')
    dt = fp.read().split('\n')[:-1]
    fp.close()

    ecg_metadata['n_leads'] = int(dt[0].split(' ')[1])
    ecg_metadata['sampling_freq'] = int(dt[0].split(' ')[2])
    ecg_metadata['signals'] = []

    for i in range(1,ecg_metadata['n_leads']+1):
        lead_data = dt[i].split(' ')
        bits_written = lead_data[1]
        adc_unit = lead_data[2]
        adc_resolution = lead_data[3]
        baseline_val = lead_data[4]
        lead_name = lead_data[-1]

        ecg_metadata['signals'].append({'lead_name':lead_name,
                                        'bits_written':bits_written,
                                        'adc_unit':adc_unit,
                                        'adc_resolution':adc_resolution,
                                        'baseline_val':baseline_val
                                        })


    for i in range(ecg_metadata['n_leads']+1,len(dt)):
        tag = dt[i].split(':')[0].split(' ')[-1]
        val = dt[i].split(':')[1].strip()

        ecg_metadata[tag] = val
        #ecg_metadata['age'] =dt[-6].split(':')[1].strip()
        #ecg_metadata['sex'] =dt[-5].split(':')[1].strip()
        #ecg_metadata['dx'] = dt[-4][5:].strip()
        #ecg_metadata['rx'] = dt[-3][5:].strip()
        #ecg_metadata['hx'] = dt[-2][5:].strip()
        #ecg_metadata['sx'] = dt[-1][5:].strip()

    if(prnt_opt):
        for ky in ecg_metadata:
            print(ky,ecg_metadata[ky])

    return ecg_metadata
