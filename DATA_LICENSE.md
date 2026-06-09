# Data Responsibilities

This repository contains code. Users provide their own CSV inputs and are responsible for ensuring they have the rights to use those inputs and any outputs generated from them.

The Apache-2.0 license applies to the project code. It does not grant rights to datasets a user may supply from outside this repository.

Historical S&P 500 constituent snapshots are loaded by default from fja05680/sp500: https://github.com/fja05680/sp500

That upstream repository describes itself as current and historical S&P 500 component lists since 1996 and is published under the MIT license.

Do not commit generated market-data outputs or input datasets unless they are synthetic, user-created, or clearly licensed for redistribution. The project code can generate reports from user-supplied CSVs, but users remain responsible for the rights attached to those inputs and derived outputs.
