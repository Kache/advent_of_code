require 'bundler/inline'

gemfile do
  source 'https://rubygems.org'
  ruby '~> 2.6.3'
  gem 'pry-byebug'
  gem 'activesupport'
  gem 'http'
  gem 'nokogiri'
end

years = 1960..2018

url = 'https://www.ssa.gov/cgi-bin/namesbystate.cgi'
xpath = '/html/body/table[2]/tbody/tr/td[2]/table/tr[./td]/td[position()=4 or position()=5]'

names = {}

years.reverse_each do |year|
  Thread.new do
    request = HTTP.post(url, body: URI.encode_www_form({ state: 'CA', year: year }))
    page = Nokogiri.HTML(request.body.to_s)
    names[year] = page.xpath(xpath).each_slice(2).map do |name, count|
      [name.text, count.text.scan(/\d/).join.to_i]
    end.to_h
  end
  sleep(0.1)
end


require 'csv'

all_names = names.each_value.inject({}, :merge).keys.sort
year_list = names.keys.sort

csv_str = CSV.generate(col_sep: "\t") do |csv|
  csv << year_list

  all_names.each do |name|

    counts = year_list.map do |year|
      names.fetch(year, {}).fetch(name, 0)
    end
    csv << [name, *counts]
  end
end; nil

`echo '#{csv_str}' | pbcopy`
