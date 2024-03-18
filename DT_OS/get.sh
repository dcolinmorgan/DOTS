curl -X GET "https://louie_armstrong:peach-Jam-42-prt@search-opensearch-dev-domain-7grknmmmm7nikv5vwklw7r4pqq.us-east-1.es.amazonaws.com/emergency-management-news/_search" -H 'Content-Type: application/json' -d '{
"_source": ["metadata.GDELT_DATE", "metadata.page_title","metadata.DocumentIdentifier", "metadata.Organizations","metadata.Persons","metadata.Themes","metadata.text", "metadata.Locations"],
    "query": {
    "match_all": {}
    }
}' | jq . > data.json
