"""
Data loader

Load dataset and split into train and test sets.
Apply patches if specified in the configuration.
"""

import pandas as pd


class DataLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.date_col = self.cfg.data.date_column
        self.target_col = self.cfg.data.target_column
        self.entity_col = getattr(
            self.cfg.data,
            "entity_column",
            getattr(self.cfg.data, "country_column", "Country"),
        )

    def patch_column(self, df, column, target_entity, source_entity):
        """
        Generic method to patch missing or incorrect data for a specific column and entity
        using data from another entity.
        """
        if column not in df.columns:
            return df

        source_series = df[df[self.entity_col] == source_entity].set_index(
            self.date_col
        )[column]
        target_mask = df[self.entity_col] == target_entity

        # Apply the mapping based on date
        df.loc[target_mask, column] = df.loc[target_mask, self.date_col].map(
            source_series
        )
        return df

    def validate_schema(self, df):
        """
        Validates only the absolutely required structural columns based on configuration.
        """
        required = [self.date_col, self.entity_col, self.target_col]
        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(f"Dataset missing required structural columns: {missing}")

        for col in (self.date_col, self.entity_col):
            n_null = df[col].isna().sum()
            if n_null > 0:
                raise ValueError(
                    f"Column '{col}' has {n_null} null values — check dataset integrity"
                )

    def load_dataset(self):
        """
        Loads the dataset, applies generic patches if configured, and returns the sorted DataFrame.
        """
        df = pd.read_csv(self.cfg.data.dataset_path, parse_dates=[self.date_col])
        df = df.sort_values([self.entity_col, self.date_col]).reset_index(drop=True)

        # Apply any patches specified in the configuration
        patches = getattr(self.cfg.data, "patches", [])
        for patch in patches:
            df = self.patch_column(
                df,
                column=getattr(patch, "column", None),
                target_entity=getattr(patch, "target_entity", None),
                source_entity=getattr(patch, "source_entity", None),
            )

        return df

    def split_train_test(self, df):
        """
        Splits the dataset according to the configuration windows and entity overrides.
        """
        train_end = pd.Timestamp(self.cfg.data.train_end)
        test_start = pd.Timestamp(self.cfg.data.test_start)

        train_df = df[df[self.date_col] <= train_end].reset_index(drop=True)

        entity_start_dates = getattr(self.cfg.data, "entity_start_dates", None)

        # Fallback logic if entity_start_dates is missing but legacy ones exist
        if entity_start_dates is None and hasattr(self.cfg.data, "ua_train_start"):
            entity_start_dates = {
                "UA": self.cfg.data.ua_train_start,
                "LT": self.cfg.data.lt_lv_train_start,
                "LV": getattr(self.cfg.data, "lt_lv_train_start", None),
            }

        if entity_start_dates:
            rows = []
            for entity, start_date in entity_start_dates.items():
                if not start_date:
                    continue
                mask = (train_df[self.entity_col] == entity) & (
                    train_df[self.date_col] >= pd.Timestamp(start_date)
                )
                rows.append(train_df[mask])

            # Include entities not specified in entity_start_dates
            specified_entities = set(entity_start_dates.keys())
            mask_other = ~train_df[self.entity_col].isin(specified_entities)
            rows.append(train_df[mask_other])

            train_df = (
                pd.concat(rows, ignore_index=True)
                .sort_values([self.entity_col, self.date_col])
                .reset_index(drop=True)
            )

        test_df = df[df[self.date_col] >= test_start]

        test_end = getattr(self.cfg.data, "test_end", None)
        if test_end:
            test_df = test_df[test_df[self.date_col] <= pd.Timestamp(test_end)]

        test_entities = getattr(
            self.cfg.data,
            "test_entities",
            getattr(self.cfg.data, "test_countries", None),
        )
        if test_entities:
            test_df = test_df[test_df[self.entity_col].isin(test_entities)]

        return train_df, test_df.reset_index(drop=True)
