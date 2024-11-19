

    @staticmethod
    def read_bounding_boxes(txt_file):
        try:
            df = pd.read_csv(txt_file, sep='\s+', names=['Class', 'Confidence', 'xmin', 'ymin', 'xmax', 'ymax'], skiprows=1)
            if {'xmin', 'ymin', 'xmax', 'ymax'}.issubset(df.columns):
                return df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
        except Exception as e:
            print(f"Error reading {txt_file}: {e}")
        return []



